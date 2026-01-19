#include "rust_retro_gamedata.h"
#include "data.h"
#include <stdlib.h>
#include <string>

/////////////////////////////////////
///////////Create/Free///////////////
/////////////////////////////////////

RetroGameData* gamedata_new() {
    RetroGameData* h = new RetroGameData;
    h->data = new Retro::GameData();
    h->scenario = new Retro::Scenario(*h->data);
    return h;
}

RetroActionSet* gamedata_valid_actions(RetroGameData* h, size_t* num_entries) {
    const auto& actions = h->scenario->validActions();

    *num_entries = actions.size();
    RetroActionSet* result = (RetroActionSet*)malloc(sizeof(RetroActionSet) * actions.size());

    size_t i = 0;
    for (const auto& [key, value_set] : actions) {
        result[i].key = key;
        result[i].num_values = value_set.size();
        result[i].values = (int*)malloc(sizeof(int) * value_set.size());

        size_t j = 0;
        for (int v : value_set) {
            result[i].values[j++] = v;
        }
        i++;
    }

    return result;
}

void gamedata_free_valid_actions(RetroActionSet* actions, size_t num_entries) {
    for (size_t i = 0; i < num_entries; i++) {
        free(actions[i].values);
    }
    free(actions);
}



/////////////////////////////////////
/////////////Methods/////////////////
/////////////////////////////////////

bool gamedata_load(RetroGameData* h, const char* data_path, const char* scenario_path) {
    bool success = true;
    success = success && h->data->load(data_path);
    success = success && h->scenario->load(scenario_path);

    return true;
}

void gamedata_reset(RetroGameData* h) {
    h->scenario->restart();
    h->scenario->reloadScripts();
}

void gamedata_update_ram(RetroGameData* h) {
    h->data->updateRam();
    h->scenario->update();
}

float gamedata_current_reward(RetroGameData* h) {
    return h->scenario->currentReward(0);
}

float gamedata_total_reward(RetroGameData* h) {
    return h->scenario->totalReward(0);
}

bool gamedata_is_done(RetroGameData* h) {
    return h->scenario->isDone();
}