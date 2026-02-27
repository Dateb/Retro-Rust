#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle
typedef struct RetroMovie RetroMovie;
struct RetroEmulator;

/////////////////////////////////////
///////////Create/Free///////////////
/////////////////////////////////////

RetroMovie* movie_new(const char* name);
void movie_close(RetroMovie* handle);

/////////////////////////////////////
/////////////Methods/////////////////
/////////////////////////////////////

bool movie_step(RetroMovie* handle);
void movie_set_key(RetroMovie* handle, int key, bool set);
void movie_configure(RetroMovie* movie_handle, RetroEmulator* emulator_handle, const char* name);
void movie_set_state(RetroMovie* handle, const uint8_t* data, size_t size);

#ifdef __cplusplus
}
#endif
