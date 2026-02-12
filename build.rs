fn main() {
    // Rebuild triggers
    println!("cargo:rerun-if-changed=c_wrapper/rust_retro_emulator.cpp");
    println!("cargo:rerun-if-changed=c_wrapper/rust_retro_emulator.h");
    println!("cargo:rerun-if-changed=c_wrapper/rust_retro_gamedata.cpp");
    println!("cargo:rerun-if-changed=c_wrapper/rust_retro_gamedata.h");
    println!("cargo:rerun-if-changed=c_wrapper/rust_retro_movie.cpp");
    println!("cargo:rerun-if-changed=c_wrapper/rust_retro_movie.h");

    println!("cargo:rerun-if-changed=third_party/libzip");

    // =========================
    // 1) Build libzip via CMake
    // =========================
    let libzip_dst = cmake::Config::new("third_party/libzip")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("ENABLE_BZIP2", "OFF")
        .define("ENABLE_LZMA", "OFF")
        .define("ENABLE_ZSTD", "OFF")
        .define("ENABLE_OPENSSL", "OFF")
        .build();

    let libzip_include = libzip_dst.join("include");
    let libzip_lib = libzip_dst.join("lib");

    println!("cargo:rustc-link-search=native={}", libzip_lib.display());
    println!("cargo:rustc-link-lib=static=zip");
    println!("cargo:rustc-link-lib=bz2");
    println!("cargo:rustc-link-lib=z");
    println!("cargo:rustc-link-lib=stdc++");

    // =========================
    // 2) Link existing libs
    // =========================
    println!("cargo:rustc-link-search=native=.");
    println!("cargo:rustc-link-lib=static=retro-base");
    println!("cargo:rustc-link-lib=dylib=lua5.1");

    // =========================
    // 3) Build C++ wrappers
    // =========================
    cc::Build::new()
        .cpp(true)
        .file("c_wrapper/rust_retro_emulator.cpp")
        .file("c_wrapper/rust_retro_gamedata.cpp")
        .file("c_wrapper/rust_retro_movie.cpp")
        .include("includes")
        .include(&libzip_include)
        .flag_if_supported("-std=c++17")
        .compile("rust_retro");
}
