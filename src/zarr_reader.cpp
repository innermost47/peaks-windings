// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Anthony Charretier
#include "zarr_reader.hpp"

#include "parallel.hpp"

#include <blosc.h>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#ifdef PW_HAVE_ZSTD
#include <zstd.h>
#endif
#ifdef PW_HAVE_ZLIB
#include <zlib.h>
#endif

#include <chrono>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace pw
{
    namespace
    {

        using json = nlohmann::json;

        // ------------------------------------------------------------------------------------------- stores
        // A key-value store: the content of a key, or nothing when the key does not exist.
        class Store
        {
        public:
            virtual ~Store() = default;
            virtual std::vector<std::optional<std::string>> get_many(const std::vector<std::string> &keys) = 0;
            std::optional<std::string> get(const std::string &key) { return std::move(get_many({key})[0]); }
        };

        // A zarr store in a local directory; the keys of one request are read in parallel by `readers` threads.
        class LocalStore : public Store
        {
        public:
            LocalStore(const std::string &root, int readers) : root_(std::filesystem::u8path(root)), readers_(std::max(1, readers))
            {
                std::error_code ec;
                if (!std::filesystem::is_directory(root_, ec))
                    throw std::runtime_error("no such zarr directory: " + root);
            }

            std::vector<std::optional<std::string>> get_many(const std::vector<std::string> &keys) override
            {
                std::vector<std::optional<std::string>> out(keys.size());
                parallel_for(static_cast<int>(keys.size()), readers_, [&](int i, int)
                             { out[i] = read_file(root_ / std::filesystem::u8path(keys[i])); });
                return out;
            }

        private:
            // The whole file, or nothing when it does not exist (a missing chunk).
            static std::optional<std::string> read_file(const std::filesystem::path &path)
            {
                std::ifstream f(path, std::ios::binary | std::ios::ate);
                if (!f)
                    return std::nullopt;
                const std::streamsize size = f.tellg();
                std::string data(static_cast<size_t>(std::max<std::streamsize>(size, 0)), '\0');
                f.seekg(0);
                if (size > 0 && !f.read(&data[0], size))
                    throw std::runtime_error("could not read " + path.u8string());
                return data;
            }

            std::filesystem::path root_;
            int readers_;
        };

        size_t write_body(char *data, size_t size, size_t count, void *user)
        {
            static_cast<std::string *>(user)->append(data, size * count);
            return size * count;
        }

        class HttpStore : public Store
        {
        public:
            HttpStore(std::string root, int connections) : root_(std::move(root)), connections_(std::max(1, connections))
            {
                static const bool initialised = curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK;
                if (!initialised)
                    throw std::runtime_error("libcurl could not be initialised");
                multi_ = curl_multi_init();
                if (!multi_)
                    throw std::runtime_error("libcurl: no multi handle");
                curl_multi_setopt(multi_, CURLMOPT_MAX_TOTAL_CONNECTIONS, static_cast<long>(connections_));
            }
            ~HttpStore() override { curl_multi_cleanup(multi_); }

            std::vector<std::optional<std::string>> get_many(const std::vector<std::string> &keys) override
            {
                struct Transfer
                {
                    size_t index = 0;
                    int attempts = 0;
                    std::string body;
                };
                std::vector<std::optional<std::string>> out(keys.size());
                std::deque<Transfer *> waiting;
                std::vector<Transfer> transfers(keys.size());
                for (size_t i = 0; i < keys.size(); ++i)
                {
                    transfers[i].index = i;
                    waiting.push_back(&transfers[i]);
                }
                const int max_attempts = 6;
                int active = 0;
                while (!waiting.empty() || active > 0)
                {
                    while (!waiting.empty() && active < connections_)
                    {
                        Transfer *t = waiting.front();
                        waiting.pop_front();
                        t->body.clear();
                        ++t->attempts;
                        CURL *easy = curl_easy_init();
                        if (!easy)
                            throw std::runtime_error("libcurl: no easy handle");
                        const std::string url = root_ + "/" + keys[t->index];
                        curl_easy_setopt(easy, CURLOPT_URL, url.c_str());
                        curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, write_body);
                        curl_easy_setopt(easy, CURLOPT_WRITEDATA, &t->body);
                        curl_easy_setopt(easy, CURLOPT_PRIVATE, t);
                        curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);
                        curl_easy_setopt(easy, CURLOPT_NOSIGNAL, 1L);
                        curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT, 30L);
                        curl_easy_setopt(easy, CURLOPT_LOW_SPEED_LIMIT, 1L);
                        curl_easy_setopt(easy, CURLOPT_LOW_SPEED_TIME, 60L);
                        curl_easy_setopt(easy, CURLOPT_USERAGENT, "peaks_windings");
                        curl_multi_add_handle(multi_, easy);
                        ++active;
                    }
                    int running = 0;
                    curl_multi_perform(multi_, &running);
                    int queued = 0;
                    while (CURLMsg *msg = curl_multi_info_read(multi_, &queued))
                    {
                        if (msg->msg != CURLMSG_DONE)
                            continue;
                        CURL *easy = msg->easy_handle;
                        const CURLcode code = msg->data.result;
                        Transfer *t = nullptr;
                        long status = 0;
                        curl_easy_getinfo(easy, CURLINFO_PRIVATE, &t);
                        curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status);
                        curl_multi_remove_handle(multi_, easy);
                        curl_easy_cleanup(easy);
                        --active;
                        const std::string &key = keys[t->index];
                        if (code == CURLE_OK && status == 200)
                        {
                            out[t->index] = std::move(t->body);
                        }
                        else if (code == CURLE_OK && (status == 404 || status == 403))
                        {
                            // S3 answers 403 for a missing key when the bucket cannot be listed: both mean "no such chunk"
                        }
                        else if (code != CURLE_OK || status >= 500 || status == 429)
                        {
                            if (t->attempts >= max_attempts)
                                throw std::runtime_error("download failed after " + std::to_string(max_attempts) +
                                                         " attempts: " + root_ + "/" + key + " (" +
                                                         (code != CURLE_OK ? curl_easy_strerror(code)
                                                                           : "http " + std::to_string(status)) +
                                                         ")");
                            // "429 Too Many Requests": wait 1, 2, 4, 8, 16 s; other failures: a short pause
                            const int pause_ms = status == 429 ? 1000 << (t->attempts - 1) : 200 * t->attempts;
                            if (status == 429)
                                std::cerr << "[warning] http 429 (too many requests): new attempt in " << pause_ms / 1000
                                          << " s (" << t->attempts << "/" << max_attempts << ")\n";
                            std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms));
                            waiting.push_back(t);
                        }
                        else
                        {
                            throw std::runtime_error("http " + std::to_string(status) + " for " + root_ + "/" + key);
                        }
                    }
                    if (active > 0)
                        curl_multi_poll(multi_, nullptr, 0, 1000, nullptr);
                }
                return out;
            }

        private:
            std::string root_;
            int connections_;
            CURLM *multi_ = nullptr;
        };

        bool starts_with(const std::string &s, const std::string &prefix) { return s.compare(0, prefix.size(), prefix) == 0; }

        std::unique_ptr<Store> make_store(std::string url, int connections)
        {
            while (!url.empty() && (url.back() == '/' || url.back() == '\\'))
                url.pop_back();
            if (starts_with(url, "s3://"))
            {
                const std::string rest = url.substr(5);
                const size_t slash = rest.find('/');
                const std::string bucket = rest.substr(0, slash);
                const std::string key = slash == std::string::npos ? "" : rest.substr(slash);
                url = "https://" + bucket + ".s3.amazonaws.com" + key;
            }
            if (starts_with(url, "http://") || starts_with(url, "https://"))
                return std::make_unique<HttpStore>(url, connections);
            if (starts_with(url, "file://"))
                url = url.substr(7);
            return std::make_unique<LocalStore>(url, connections);
        }

        // ------------------------------------------------------------------------------------------- metadata
        enum class Codec
        {
            none,
            blosc,
            zstd,
            zlib
        };

        struct Meta
        {
            std::vector<int64_t> shape, chunks;
            int cv_depth = CV_8U;
            size_t item_size = 1;
            double fill_value = 0;
            Codec codec = Codec::none;
            bool crc32c = false;      // zarr 3: a 4-byte checksum follows every chunk
            std::string key_prefix;   // "0/" ...
            std::string chunk_prefix; // "" (zarr 2, and zarr 3 "v2" keys) or "c<sep>" (zarr 3 default keys)
            std::string separator = ".";
            int version = 2;
            std::string dtype_name;
        };

        void set_dtype(Meta &meta, const std::string &name)
        {
            struct Entry
            {
                const char *v2;
                const char *v3;
                int depth;
                size_t size;
            };
            static const Entry table[] = {
                {"u1", "uint8", CV_8U, 1},
                {"i1", "int8", CV_8S, 1},
                {"u2", "uint16", CV_16U, 2},
                {"i2", "int16", CV_16S, 2},
                {"i4", "int32", CV_32S, 4},
                {"f4", "float32", CV_32F, 4},
                {"f8", "float64", CV_64F, 8},
            };
            meta.dtype_name = name;
            std::string code = name;
            if (name.size() == 3 && (name[0] == '<' || name[0] == '|' || name[0] == '=' || name[0] == '>'))
            {
                if (name[0] == '>' && name[2] != '1')
                    throw std::runtime_error("big-endian zarr arrays are not supported");
                code = name.substr(1);
            }
            for (const Entry &e : table)
                if (code == e.v2 || code == e.v3)
                {
                    meta.cv_depth = e.depth;
                    meta.item_size = e.size;
                    return;
                }
            throw std::runtime_error("unsupported zarr data type: " + name);
        }

        Codec codec_from_name(const std::string &name)
        {
            if (name == "blosc")
                return Codec::blosc;
            if (name == "zstd")
                return Codec::zstd;
            if (name == "zlib" || name == "gzip")
                return Codec::zlib;
            throw std::runtime_error("unsupported zarr compressor: " + name);
        }

        void check_3d(const Meta &meta)
        {
            if (meta.shape.size() != 3 || meta.chunks.size() != 3)
                throw std::runtime_error("a 3-D (z, y, x) zarr array is expected, got " + std::to_string(meta.shape.size()) +
                                         " dimensions");
            for (int64_t c : meta.chunks)
                if (c <= 0)
                    throw std::runtime_error("invalid zarr chunk shape");
        }

        Meta parse_v2(const json &j, const std::string &prefix)
        {
            Meta meta;
            meta.version = 2;
            meta.key_prefix = prefix;
            meta.shape = j.at("shape").get<std::vector<int64_t>>();
            meta.chunks = j.at("chunks").get<std::vector<int64_t>>();
            set_dtype(meta, j.at("dtype").get<std::string>());
            if (j.value("order", std::string("C")) != "C")
                throw std::runtime_error("zarr arrays in F order are not supported");
            if (j.contains("filters") && !j["filters"].is_null() && !j["filters"].empty())
                throw std::runtime_error("zarr filters are not supported");
            if (j.contains("compressor") && !j["compressor"].is_null())
                meta.codec = codec_from_name(j["compressor"].at("id").get<std::string>());
            if (j.contains("fill_value") && j["fill_value"].is_number())
                meta.fill_value = j["fill_value"].get<double>();
            if (j.contains("dimension_separator") && j["dimension_separator"].is_string())
                meta.separator = j["dimension_separator"].get<std::string>();
            check_3d(meta);
            return meta;
        }

        Meta parse_v3(const json &j, const std::string &prefix)
        {
            Meta meta;
            meta.version = 3;
            meta.key_prefix = prefix;
            meta.shape = j.at("shape").get<std::vector<int64_t>>();
            const json &grid = j.at("chunk_grid");
            if (grid.at("name") != "regular")
                throw std::runtime_error("only regular zarr chunk grids are supported");
            meta.chunks = grid.at("configuration").at("chunk_shape").get<std::vector<int64_t>>();
            set_dtype(meta, j.at("data_type").get<std::string>());
            meta.separator = "/";
            meta.chunk_prefix = "c/";
            if (j.contains("chunk_key_encoding"))
            {
                const json &enc = j["chunk_key_encoding"];
                const std::string name = enc.value("name", std::string("default"));
                const json conf = enc.value("configuration", json::object());
                if (name == "v2")
                {
                    meta.separator = conf.value("separator", std::string("."));
                    meta.chunk_prefix = "";
                }
                else
                {
                    meta.separator = conf.value("separator", std::string("/"));
                    meta.chunk_prefix = "c" + meta.separator;
                }
            }
            if (j.contains("fill_value") && j["fill_value"].is_number())
                meta.fill_value = j["fill_value"].get<double>();
            for (const json &codec : j.value("codecs", json::array()))
            {
                const std::string name = codec.at("name").get<std::string>();
                const json conf = codec.value("configuration", json::object());
                if (name == "bytes")
                {
                    if (meta.item_size > 1 && conf.value("endian", std::string("little")) != "little")
                        throw std::runtime_error("big-endian zarr arrays are not supported");
                }
                else if (name == "crc32c")
                {
                    meta.crc32c = true;
                }
                else if (name == "sharding_indexed")
                {
                    throw std::runtime_error("sharded zarr 3 arrays are not supported by this reader");
                }
                else if (name == "transpose")
                {
                    throw std::runtime_error("the zarr 3 transpose codec is not supported by this reader");
                }
                else
                {
                    if (meta.codec != Codec::none)
                        throw std::runtime_error("several zarr compressors in a row are not supported");
                    meta.codec = codec_from_name(name);
                }
            }
            check_3d(meta);
            return meta;
        }

        // The array at `prefix` ("" or "name/"), if there is one. `is_group` tells a zarr 3 group was found there.
        std::optional<Meta> find_array(Store &store, const std::string &prefix)
        {
            if (auto text = store.get(prefix + ".zarray"))
                return parse_v2(json::parse(*text), prefix);
            if (auto text = store.get(prefix + "zarr.json"))
            {
                const json j = json::parse(*text);
                if (j.value("node_type", std::string()) == "array")
                    return parse_v3(j, prefix);
            }
            return std::nullopt;
        }

        // ------------------------------------------------------------------------------------------- decoding
        size_t blosc_decoded_size(const std::string &chunk)
        {
            if (chunk.size() < BLOSC_MIN_HEADER_LENGTH)
                throw std::runtime_error("a blosc chunk is truncated");
            size_t nbytes = 0, cbytes = 0, blocksize = 0;
            blosc_cbuffer_sizes(chunk.data(), &nbytes, &cbytes, &blocksize);
            if (cbytes > chunk.size())
                throw std::runtime_error("a blosc chunk is truncated");
            return nbytes;
        }

        // The whole chunk, decoded into `out` (which must hold `expected` bytes).
        void decode_chunk(const Meta &meta, const std::string &chunk, unsigned char *out, size_t expected)
        {
            const size_t size = chunk.size() - (meta.crc32c ? std::min<size_t>(4, chunk.size()) : 0);
            switch (meta.codec)
            {
            case Codec::none:
                if (size != expected)
                    throw std::runtime_error("an uncompressed zarr chunk has the wrong size");
                std::memcpy(out, chunk.data(), expected);
                return;
            case Codec::blosc:
            {
                if (blosc_decoded_size(chunk) != expected)
                    throw std::runtime_error("a blosc chunk has the wrong size");
                if (blosc_decompress_ctx(chunk.data(), out, expected, 1) != static_cast<int>(expected))
                    throw std::runtime_error("blosc could not decode a chunk");
                return;
            }
            case Codec::zstd:
            {
#ifdef PW_HAVE_ZSTD
                const size_t n = ZSTD_decompress(out, expected, chunk.data(), size);
                if (ZSTD_isError(n) || n != expected)
                    throw std::runtime_error("zstd could not decode a chunk");
                return;
#else
                throw std::runtime_error("this build has no zstd support (zstd was not found by CMake)");
#endif
            }
            case Codec::zlib:
            {
#ifdef PW_HAVE_ZLIB
                z_stream zs;
                std::memset(&zs, 0, sizeof(zs));
                if (inflateInit2(&zs, 15 + 32) != Z_OK)
                    throw std::runtime_error("zlib could not start"); // zlib or gzip
                zs.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(chunk.data()));
                zs.avail_in = static_cast<uInt>(size);
                zs.next_out = out;
                zs.avail_out = static_cast<uInt>(expected);
                const int rc = inflate(&zs, Z_FINISH);
                const size_t n = zs.total_out;
                inflateEnd(&zs);
                if (rc != Z_STREAM_END || n != expected)
                    throw std::runtime_error("zlib could not decode a chunk");
                return;
#else
                throw std::runtime_error("this build has no zlib support (zlib was not found by CMake)");
#endif
            }
            }
        }

        // ------------------------------------------------------------------------------------------- the volume
        class Volume : public ZarrVolume
        {
        public:
            Volume(std::unique_ptr<Store> store, Meta meta, std::string url, int threads, double cache_gb)
                : store_(std::move(store)), meta_(std::move(meta)), url_(std::move(url)), threads_(threads)
            {
                decoded_budget_ = static_cast<size_t>(std::max(cache_gb, 0.0) * 1024.0 * 1024.0 * 1024.0);
                ny_ = static_cast<int>((meta_.shape[1] + meta_.chunks[1] - 1) / meta_.chunks[1]);
                nx_ = static_cast<int>((meta_.shape[2] + meta_.chunks[2] - 1) / meta_.chunks[2]);
                plane_bytes_ = static_cast<size_t>(meta_.chunks[1]) * meta_.chunks[2] * meta_.item_size;
                chunk_bytes_ = plane_bytes_ * meta_.chunks[0];
            }

            const std::vector<int64_t> &shape() const override { return meta_.shape; }
            const std::vector<int64_t> &chunks() const override { return meta_.chunks; }
            int cv_type() const override { return CV_MAKETYPE(meta_.cv_depth, 1); }

            std::string description() const override
            {
                static const char *names[] = {"none", "blosc", "zstd", "zlib"};
                std::ostringstream s;
                s << "zarr " << meta_.version << " array '" << meta_.key_prefix << "', shape (" << meta_.shape[0] << ", "
                  << meta_.shape[1] << ", " << meta_.shape[2] << "), chunks (" << meta_.chunks[0] << ", " << meta_.chunks[1]
                  << ", " << meta_.chunks[2] << "), " << meta_.dtype_name << ", compressor "
                  << names[static_cast<int>(meta_.codec)];
                return s.str();
            }

            cv::Mat read_slice(int64_t z) override
            {
                if (z < 0 || z >= meta_.shape[0])
                    throw std::out_of_range("slice " + std::to_string(z) + " is outside the volume (" +
                                            std::to_string(meta_.shape[0]) + " slices)");
                const int64_t plane = z / meta_.chunks[0];
                const size_t z_local = static_cast<size_t>(z % meta_.chunks[0]);
                if (plane != cached_plane_)
                    load_plane(plane);

                const int h = static_cast<int>(meta_.shape[1]), w = static_cast<int>(meta_.shape[2]);
                cv::Mat out(h, w, CV_MAKETYPE(meta_.cv_depth, 1), cv::Scalar(meta_.fill_value));
                const int threads = threads_ > 0 ? threads_ : hardware_threads();
                std::vector<std::vector<unsigned char>> scratch(threads);
                const size_t offset = z_local * plane_bytes_;
                parallel_for(ny_ * nx_, threads, [&](int i, int worker)
                             {
            const std::optional<std::string>& chunk = chunks_[i];
            if (!chunk) return;  // a missing chunk is all fill value
            const unsigned char* plane_data = nullptr;
            if (decoded_) {
                plane_data = reinterpret_cast<const unsigned char*>(chunk->data()) + offset;
            } else {
                std::vector<unsigned char>& buf = scratch[worker];
                plane_data = decode_plane(*chunk, offset, buf);
            }
            const int cy = i / nx_, cx = i % nx_;
            const int64_t y0 = cy * meta_.chunks[1], x0 = cx * meta_.chunks[2];
            const int rows = static_cast<int>(std::min<int64_t>(meta_.chunks[1], h - y0));
            const size_t row_bytes = static_cast<size_t>(std::min<int64_t>(meta_.chunks[2], w - x0)) * meta_.item_size;
            const size_t stride = static_cast<size_t>(meta_.chunks[2]) * meta_.item_size;
            for (int r = 0; r < rows; ++r)
                std::memcpy(out.ptr(static_cast<int>(y0) + r) + x0 * meta_.item_size, plane_data + r * stride,
                            row_bytes); });
                return out;
            }

            std::vector<cv::Mat> read_region(int64_t z0, int64_t z1, int64_t y0, int64_t y1, int64_t x0,
                                             int64_t x1) override
            {
                z0 = std::max<int64_t>(z0, 0), y0 = std::max<int64_t>(y0, 0), x0 = std::max<int64_t>(x0, 0);
                z1 = std::min(z1, meta_.shape[0]), y1 = std::min(y1, meta_.shape[1]), x1 = std::min(x1, meta_.shape[2]);
                std::vector<cv::Mat> out;
                if (z1 <= z0)
                    return out;
                const int h = static_cast<int>(std::max<int64_t>(y1 - y0, 0)), w = static_cast<int>(std::max<int64_t>(x1 - x0, 0));
                for (int64_t z = z0; z < z1; ++z)
                    out.emplace_back(h, w, CV_MAKETYPE(meta_.cv_depth, 1), cv::Scalar(meta_.fill_value));
                if (h == 0 || w == 0)
                    return out;

                struct Wanted
                {
                    int64_t cz, cy, cx;
                    std::string key;
                };
                std::vector<Wanted> wanted;
                for (int64_t cz = z0 / meta_.chunks[0]; cz <= (z1 - 1) / meta_.chunks[0]; ++cz)
                    for (int64_t cy = y0 / meta_.chunks[1]; cy <= (y1 - 1) / meta_.chunks[1]; ++cy)
                        for (int64_t cx = x0 / meta_.chunks[2]; cx <= (x1 - 1) / meta_.chunks[2]; ++cx)
                            wanted.push_back({cz, cy, cx, chunk_key(cz, static_cast<int>(cy), static_cast<int>(cx))});

                std::vector<std::string> missing;
                for (const Wanted &c : wanted)
                    if (!region_cache_.count(c.key))
                        missing.push_back(c.key);
                if (!missing.empty())
                {
                    std::vector<std::optional<std::string>> fetched = store_->get_many(missing);
                    size_t incoming = 0;
                    for (const auto &f : fetched)
                        incoming += f ? f->size() : 0;
                    if (region_cache_bytes_ + incoming > region_cache_budget_)
                    {
                        // drop what this region does not need
                        for (auto it = region_cache_.begin(); it != region_cache_.end();)
                        {
                            bool needed = false;
                            for (const Wanted &c : wanted)
                                needed = needed || c.key == it->first;
                            if (needed)
                            {
                                ++it;
                            }
                            else
                            {
                                region_cache_bytes_ -= it->second ? it->second->size() : 0;
                                it = region_cache_.erase(it);
                            }
                        }
                    }
                    for (size_t i = 0; i < missing.size(); ++i)
                    {
                        region_cache_bytes_ += fetched[i] ? fetched[i]->size() : 0;
                        region_cache_[missing[i]] = std::move(fetched[i]);
                    }
                }

                const int threads = threads_ > 0 ? threads_ : hardware_threads();
                std::vector<std::vector<unsigned char>> scratch(threads);
                const size_t item = meta_.item_size;
                parallel_for(static_cast<int>(wanted.size()), threads, [&](int i, int worker)
                             {
            const Wanted& c = wanted[i];
            const std::optional<std::string>& chunk = region_cache_.at(c.key);
            if (!chunk) return;  // a missing chunk is all fill value
            std::vector<unsigned char>& buf = scratch[worker];
            buf.resize(chunk_bytes_);
            decode_chunk(meta_, *chunk, buf.data(), chunk_bytes_);
            // the part of this chunk inside the region, in volume coordinates
            const int64_t oz = c.cz * meta_.chunks[0], oy = c.cy * meta_.chunks[1], ox = c.cx * meta_.chunks[2];
            const int64_t za = std::max(z0, oz), zb = std::min(z1, oz + meta_.chunks[0]);
            const int64_t ya = std::max(y0, oy), yb = std::min(y1, oy + meta_.chunks[1]);
            const int64_t xa = std::max(x0, ox), xb = std::min(x1, ox + meta_.chunks[2]);
            if (xb <= xa) return;
            const size_t row_stride = static_cast<size_t>(meta_.chunks[2]) * item;
            for (int64_t z = za; z < zb; ++z)
                for (int64_t y = ya; y < yb; ++y) {
                    const unsigned char* src =
                        buf.data() + (z - oz) * plane_bytes_ + (y - oy) * row_stride + (xa - ox) * item;
                    std::memcpy(out[z - z0].ptr(static_cast<int>(y - y0)) + (xa - x0) * item, src, (xb - xa) * item);
                } });
                return out;
            }

            std::vector<cv::Mat> read_chunks(const std::vector<ChunkIndex> &chunks) override
            {
                std::vector<std::string> keys;
                keys.reserve(chunks.size());
                for (const ChunkIndex &c : chunks)
                {
                    if (c.z < 0 || c.y < 0 || c.x < 0 || c.z * meta_.chunks[0] >= meta_.shape[0] || c.y >= ny_ ||
                        c.x >= nx_)
                        throw std::out_of_range("chunk (" + std::to_string(c.z) + ", " + std::to_string(c.y) + ", " +
                                                std::to_string(c.x) + ") is outside the chunk grid");
                    keys.push_back(chunk_key(c.z, static_cast<int>(c.y), static_cast<int>(c.x)));
                }
                std::vector<std::optional<std::string>> fetched = store_->get_many(keys);

                const int sizes[3] = {static_cast<int>(meta_.chunks[0]), static_cast<int>(meta_.chunks[1]),
                                      static_cast<int>(meta_.chunks[2])};
                std::vector<cv::Mat> out(chunks.size());
                const int threads = threads_ > 0 ? threads_ : hardware_threads();
                parallel_for(static_cast<int>(chunks.size()), threads, [&](int i, int)
                             {
                cv::Mat& m = out[i];
                if (!fetched[i]) {
                    m = cv::Mat(3, sizes, CV_MAKETYPE(meta_.cv_depth, 1), cv::Scalar(meta_.fill_value));
                    return;
                }
                m.create(3, sizes, CV_MAKETYPE(meta_.cv_depth, 1));
                decode_chunk(meta_, *fetched[i], m.data, chunk_bytes_);
                std::string().swap(*fetched[i]); });
                return out;
            }

        private:
            std::string chunk_key(int64_t cz, int cy, int cx) const
            {
                return meta_.key_prefix + meta_.chunk_prefix + std::to_string(cz) + meta_.separator + std::to_string(cy) +
                       meta_.separator + std::to_string(cx);
            }

            // Downloads (or reads) every chunk of one chunk plane. When the decoded plane fits the budget it is decoded
            // once; otherwise the compressed chunks are kept and every slice decodes what it needs.
            void load_plane(int64_t plane)
            {
                cached_plane_ = -1;
                chunks_.clear();
                std::vector<std::string> keys;
                keys.reserve(static_cast<size_t>(ny_) * nx_);
                for (int cy = 0; cy < ny_; ++cy)
                    for (int cx = 0; cx < nx_; ++cx)
                        keys.push_back(chunk_key(plane, cy, cx));
                chunks_ = store_->get_many(keys);

                const size_t decoded_total = chunk_bytes_ * chunks_.size();
                decoded_ = meta_.codec == Codec::none || decoded_total <= decoded_budget_;
                if (decoded_ && meta_.codec != Codec::none)
                {
                    const int threads = threads_ > 0 ? threads_ : hardware_threads();
                    parallel_for(static_cast<int>(chunks_.size()), threads, [&](int i, int)
                                 {
                if (!chunks_[i]) return;
                std::string raw(chunk_bytes_, '\0');
                decode_chunk(meta_, *chunks_[i], reinterpret_cast<unsigned char*>(&raw[0]), chunk_bytes_);
                chunks_[i] = std::move(raw); });
                }
                else if (decoded_)
                {
                    for (const auto &chunk : chunks_)
                        if (chunk && chunk->size() < chunk_bytes_)
                            throw std::runtime_error("a zarr chunk is truncated");
                }
                cached_plane_ = plane;
            }

            // One z plane of a compressed chunk. Blosc decodes only the blocks that hold the plane.
            const unsigned char *decode_plane(const std::string &chunk, size_t offset, std::vector<unsigned char> &buf) const
            {
                if (meta_.codec == Codec::blosc)
                {
                    if (blosc_decoded_size(chunk) != chunk_bytes_)
                        throw std::runtime_error("a blosc chunk has the wrong size");
                    size_t typesize = 1;
                    int flags = 0;
                    blosc_cbuffer_metainfo(chunk.data(), &typesize, &flags);
                    if (typesize > 0 && offset % typesize == 0 && plane_bytes_ % typesize == 0)
                    {
                        buf.resize(plane_bytes_);
                        const int n = blosc_getitem(chunk.data(), static_cast<int>(offset / typesize),
                                                    static_cast<int>(plane_bytes_ / typesize), buf.data());
                        if (n != static_cast<int>(plane_bytes_))
                            throw std::runtime_error("blosc could not decode a chunk");
                        return buf.data();
                    }
                }
                buf.resize(chunk_bytes_);
                decode_chunk(meta_, chunk, buf.data(), chunk_bytes_);
                return buf.data() + offset;
            }

            std::unique_ptr<Store> store_;
            Meta meta_;
            std::string url_;
            int threads_ = 0;
            int ny_ = 0, nx_ = 0;
            size_t plane_bytes_ = 0, chunk_bytes_ = 0;
            size_t decoded_budget_ = 0; // a decoded chunk plane is kept in memory up to this many bytes
            int64_t cached_plane_ = -1;
            bool decoded_ = false;
            std::vector<std::optional<std::string>> chunks_;
            // compressed chunks kept between read_region calls
            std::map<std::string, std::optional<std::string>> region_cache_;
            size_t region_cache_bytes_ = 0;
            size_t region_cache_budget_ = size_t(1) << 30;
        };

    } // namespace

    std::unique_ptr<ZarrVolume> ZarrVolume::open(const std::string &url, int connections, int threads, double cache_gb)
    {
        std::unique_ptr<Store> store = make_store(url, connections);
        // over http the keys of a group cannot be listed, so the usual array names are asked for directly
        for (const char *prefix : {"", "0/", "data/", "volume/"})
            if (std::optional<Meta> meta = find_array(*store, prefix))
                return std::make_unique<Volume>(std::move(store), std::move(*meta), url, threads, cache_gb);
        throw std::runtime_error("no zarr array found in " + url + " (tried the root, '0', 'data' and 'volume')");
    }

} // namespace pw
