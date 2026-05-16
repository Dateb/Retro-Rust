#include "rust_retro_movie.h"
#include "rust_retro_emulator.h"
#include "emulator.h"
#include <stdlib.h>
#include <string>
#include "movie.h"
#include "movie-bk2.h"

struct RetroMovie {
    std::unique_ptr<Retro::MovieBK2> movie;
};

/////////////////////////////////////
///////////Create/Free///////////////
/////////////////////////////////////

RetroMovie* movie_new(const char* name) {
    RetroMovie* handle = new RetroMovie;
    handle->movie = std::make_unique<Retro::MovieBK2>(name, true, 1);
    return handle;
}

void movie_close(RetroMovie* handle) {
    handle->movie->close();
}

/////////////////////////////////////
/////////////Methods/////////////////
/////////////////////////////////////

bool movie_step(RetroMovie* handle) {
    return handle->movie->step();
}
void movie_configure(RetroMovie* movie_handle, RetroEmulator* emulator_handle, const char* name) {
    static_cast<Retro::MovieBK2*>(movie_handle->movie.get())->setGameName(name);
    static_cast<Retro::MovieBK2*>(movie_handle->movie.get())->loadKeymap(emulator_get_core(emulator_handle));
}
void movie_set_key(RetroMovie* handle, int key, bool set) {
    handle->movie->setKey(key, set, 0);
}
void movie_set_state(RetroMovie* handle, const uint8_t* data, size_t size) {
    handle->movie->setState(const_cast<uint8_t*>(data), size);
}
