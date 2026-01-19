fn main() {
    // Rebuild whenever C wrapper changes
    println!("cargo:rerun-if-changed=c_wrapper/rust_retro_emulator.cpp");
    println!("cargo:rerun-if-changed=c_wrapper/rust_retro_emulator.h");

    println!("cargo:rerun-if-changed=c_wrapper/rust_retro_gamedata.cpp");
    println!("cargo:rerun-if-changed=c_wrapper/rust_retro_gamedata.h");

    // Tell Rust where the static library is
    println!("cargo:rustc-link-search=native=."); // folder with libretro-base.a
    println!("cargo:rustc-link-lib=static=retro-base"); // link libretro-base.a
    println!("cargo:rustc-link-lib=dylib=lua5.1");

    cc::Build::new()
        .cpp(true) // treat as C++
        .file("c_wrapper/rust_retro_emulator.cpp")
        .file("c_wrapper/rust_retro_gamedata.cpp")
        .include("includes")
        .compile("rust_retro");
}
