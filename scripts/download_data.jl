#=
After publishing your deposit on Zenodo, fill in the constants below (or pass `--record-id` / `--zip-name`).

Example:
  julia --project=. scripts/download_data.jl
  julia --project=. scripts/download_data.jl --dry-run
  julia --project=. scripts/download_data.jl --force --subdir zenodo

Zenodo: record URL is `https://zenodo.org/records/<ID>`; the API is `GET <base>/api/records/<ID>`.
=#

using DrWatson
# `@quickactivate :DRL_RDE` expands to `using DRL_RDE`, which loads the entire project package and its heavy deps.
# Activating the repo and checking the project name matches without importing the package:
quickactivate(dirname(@__DIR__), "DRL_RDE")

using ArgParse
using Downloads
using HTTP
using JSON
using Mmap
using ZipArchives: ZipReader, zip_isdir, zip_names, zip_openentry, zip_readentry, zip_test

# --- Configure after publishing (CLI flags override these defaults) ---
"""Numeric Zenodo record id from the record URL, e.g. `https://zenodo.org/records/1234567` → `1234567`."""
const DEFAULT_ZENODO_RECORD_ID = 23471100

"""Exact zip filename as shown on Zenodo (must match `key` / filename in the deposit)."""
const DEFAULT_ZENODO_ZIP_FILENAME = "data.zip"

"""`https://zenodo.org` or `https://sandbox.zenodo.org` for testing."""
const DEFAULT_ZENODO_BASE_URL = "https://zenodo.org"

function parse_commandline()
    s = ArgParseSettings(; description = "Download a Zenodo zip, verify metadata, extract to a temp dir, install into datadir.")
    @add_arg_table! s begin
        "--record-id"
        help = "Zenodo record id (overrides DEFAULT_ZENODO_RECORD_ID in this script)"
        arg_type = Int
        default = DEFAULT_ZENODO_RECORD_ID
        "--zip-name"
        help = "Zip filename on Zenodo (overrides DEFAULT_ZENODO_ZIP_FILENAME)"
        arg_type = String
        default = DEFAULT_ZENODO_ZIP_FILENAME
        "--base-url"
        help = "Zenodo base URL (production or sandbox)"
        arg_type = String
        default = DEFAULT_ZENODO_BASE_URL
        "--subdir"
        help = "Subpath under DrWatson datadir() for installed files (empty string = datadir root)"
        arg_type = String
        default = ""
        "--strip-components"
        help = "After extraction, descend this many single-directory levels before installing (like tar --strip-components)"
        arg_type = Int
        default = 0
        "--force"
        help = "Overwrite existing files in the destination when copying from the temp extract"
        action = :store_true
        "--dry-run"
        help = "Only print URLs and metadata; do not download or write data/"
        action = :store_true
    end
    return parse_args(s)
end

function zenodo_api_records_url(base::AbstractString, record_id::Int)::String
    base_stripped = rstrip(base, '/')
    return "$(base_stripped)/api/records/$(record_id)"
end

function find_file_entry(files, zip_name::AbstractString)
    for f in files
        key = get(f, "key", nothing)
        fn = get(f, "filename", nothing)
        candidate = if key isa String
            key
        elseif fn isa String
            fn
        else
            continue
        end
        candidate == zip_name && return f
    end
    return nothing
end

function file_download_url(file_entry::AbstractDict)::Union{Nothing, String}
    links = get(file_entry, "links", nothing)
    links isa AbstractDict || return nothing
    self = get(links, "self", nothing)
    self isa String && return self
    return nothing
end

function file_size_expected(file_entry::AbstractDict)::Union{Nothing, Int}
    for k in ("size", "filesize")
        v = get(file_entry, k, nothing)
        v isa Integer && return Int(v)
        v isa AbstractFloat && return Int(round(v))
    end
    return nothing
end

function file_checksum_md5_hex(file_entry::AbstractDict)::Union{Nothing, String}
    c = get(file_entry, "checksum", nothing)
    c isa String || return nothing
    if startswith(c, "md5:")
        return String(SubString(c, 5))
    end
    return nothing
end

function is_safe_extract_path(root::AbstractString, entry_name::AbstractString)::Bool
    rel = String(replace(entry_name, '\\' => '/'))
    parts = split(rel, '/'; keepempty = false)
    any(==(".."), parts) && return false
    isempty(parts) && return false
    dest = foldl((a, b) -> joinpath(a, b), parts; init = root)
    dest_a = abspath(dest)
    root_a = abspath(root)
    sep = Sys.iswindows() ? '\\' : '/'
    if dest_a == root_a
        return true
    end
    prefix = string(root_a, sep)
    return startswith(dest_a, prefix)
end

function extract_zip_to_dir(zip_path::AbstractString, dest_dir::AbstractString)::Nothing
    mkpath(dest_dir)
    sz = filesize(zip_path)
    sz ≥ 0 || error("invalid file size for $(repr(zip_path))")
    data = open(zip_path) do io
        return Mmap.mmap(io, Vector{UInt8}, (sz,))
    end
    archive = ZipReader(data)
    zip_test(archive)
    for name in zip_names(archive)
        zip_isdir(archive, name) && continue
        is_safe_extract_path(dest_dir, name) || error("unsafe zip path $(repr(name)): possible zip-slip")
        out_path = foldl((a, b) -> joinpath(a, b), split(String(replace(name, '\\' => '/')), '/'; keepempty = false); init = dest_dir)
        mkpath(dirname(out_path))
        bytes = zip_readentry(archive, name)
        write(out_path, bytes)
    end
    return nothing
end

function resolve_strip_root(extract_root::AbstractString, strip::Int)::String
    if strip ≤ 0
        return String(extract_root)
    end
    root = String(extract_root)
    for _ in 1:strip
        ents = readdir(root)
        length(ents) == 1 || error(
            "strip-components: expected exactly one entry in $(repr(root)), got $(length(ents)): $(repr(ents)). " *
                "Use a smaller --strip-components value or fix the archive layout.",
        )
        root = joinpath(root, only(ents))
        isdir(root) || error("strip-components: not a directory: $(repr(root))")
    end
    return root
end

function install_tree(src_root::AbstractString, dest_parent::AbstractString; force::Bool)::Nothing
    mkpath(dest_parent)
    for name in readdir(src_root)
        s = joinpath(src_root, name)
        d = joinpath(dest_parent, name)
        cp(s, d; force = force)
    end
    return nothing
end

function fail_exit(msg::AbstractString)::Nothing
    @error msg
    exit(1)
    return nothing
end

function main()::Nothing
    args = parse_commandline()
    record_id = args["record-id"]
    zip_name = args["zip-name"]
    base_url = args["base-url"]
    subdir = args["subdir"]
    strip_n = args["strip-components"]
    force = args["force"]
    dry = args["dry-run"]

    if record_id ≤ 0
        fail_exit(
            "Set a positive Zenodo record id: edit DEFAULT_ZENODO_RECORD_ID in this script or pass --record-id. " *
                "Find the id in your Zenodo record URL, e.g. https://zenodo.org/records/1234567 → 1234567.",
        )
    end

    api_url = zenodo_api_records_url(base_url, record_id)
    dest_dir = isempty(subdir) ? datadir() : datadir(subdir)
    if !isempty(subdir)
        @warn (
            "Installing under --subdir=$(repr(subdir)) → $(repr(dest_dir)). " *
                "Scripts in this repo assume data in the DrWatson data folder root (empty --subdir / datadir() with no extra path), not in a subdirectory. " *
                "You may need to move or symlink files so paths match what other scripts expect."
        )
    end

    resp = try
        HTTP.get(api_url)
    catch e
        fail_exit(
            "Failed to reach Zenodo API at $(repr(api_url)). " *
                "Check your network, VPN, firewall, and that Zenodo is reachable. Underlying error: $(sprint(showerror, e))",
        )
    end
    if resp.status == 404
        fail_exit(
            "HTTP 404 for $(repr(api_url)). Wrong record id or the deposit is not public. " *
                "Open the record in a browser; a new version may have a new record id.",
        )
    end
    if resp.status ≠ 200
        fail_exit(
            "Zenodo API returned HTTP $(resp.status) for $(repr(api_url)). " *
                "Check the record id and that the deposit exists.",
        )
    end

    data = try
        JSON.parse(String(resp.body))
    catch e
        fail_exit("Failed to parse JSON from Zenodo API: $(sprint(showerror, e))")
    end

    files = get(data, "files", nothing)
    if files === nothing || !(files isa AbstractVector)
        fail_exit("Unexpected Zenodo API response: missing or invalid `files` array. Response keys: $(repr(collect(keys(data))))")
    end

    file_entry = find_file_entry(files, zip_name)
    if file_entry === nothing
        avail = map(files) do f
            get(f, "key", get(f, "filename", "?"))
        end
        fail_exit(
            "No file named $(repr(zip_name)) in this deposit. Available: $(repr(avail)). " *
                "Fix DEFAULT_ZENODO_ZIP_FILENAME or pass --zip-name.",
        )
    end

    expected_size = file_size_expected(file_entry)
    dl_url = file_download_url(file_entry)
    if dl_url === nothing
        enc = HTTP.URIs.escapepath(zip_name)
        dl_url = "$(rstrip(base_url, '/'))/records/$(record_id)/files/$(enc)?download=1"
    end

    md5_hex = file_checksum_md5_hex(file_entry)
    doi = get(data, "doi", get(get(data, "metadata", Dict()), "doi", nothing))

    @info "Zenodo download" record_id doi expected_size api_url download_url = dl_url destination = dest_dir md5_expected = md5_hex

    if dry
        @info "Dry run: not downloading or extracting."
        return nothing
    end

    tmp_parent = mktempdir()
    tmp_zip = joinpath(tmp_parent, "download.zip")
    try
        try
            Downloads.download(dl_url, tmp_zip)
        catch e
            fail_exit(
                "Download failed from $(repr(dl_url)). " *
                    "Try opening the URL in a browser; check SSL/VPN/firewall. Error: $(sprint(showerror, e))",
            )
        end

        fsz = filesize(tmp_zip)
        if expected_size !== nothing && fsz ≠ expected_size
            fail_exit(
                "Downloaded file size $(fsz) does not match Zenodo metadata ($(expected_size)). " *
                    "Delete any partial file and retry; the archive may have been replaced on Zenodo.",
            )
        end

        extract_root = joinpath(tmp_parent, "extract")
        mkpath(extract_root)
        try
            extract_zip_to_dir(tmp_zip, extract_root)
        catch e
            fail_exit(
                "Zip read/extract failed (corrupt download or unsupported zip). " *
                    "Re-download or verify the archive manually. Error: $(sprint(showerror, e))",
            )
        end

        content_root = resolve_strip_root(extract_root, strip_n)

        if ispath(dest_dir) && !isempty(readdir(dest_dir)) && !force
            fail_exit(
                "Destination $(repr(dest_dir)) is not empty. Use --force to overwrite existing entries, or choose another --subdir.",
            )
        end
        mkpath(dest_dir)

        try
            install_tree(content_root, dest_dir; force = force)
        catch e
            fail_exit(
                "Failed to copy extracted files into $(repr(dest_dir)). " *
                    "Check disk space and permissions. Error: $(sprint(showerror, e))",
            )
        end

        @info "Installed Zenodo archive into" dest_dir
    finally
        rm(tmp_parent; recursive = true)
    end

    return nothing
end

main()
