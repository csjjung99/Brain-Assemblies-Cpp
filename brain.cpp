//
//  brain.cpp
//  Brain
//
//  Created by Chris Jung on 9/09/21.
//  Copyright © 2021 Chris Jung. All rights reserved.
//

#include <algorithm>
#include <iostream>

#include "brain.hpp"
#include <random>

#define float double

using namespace std;

namespace {
    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<> dist(0, 1);
}

Area NULL_AREA("NULL", 0, 0, 0, 0);
Stimulus NULL_STIM("NULL", 0, 0, 0);

static vector<vector<pair<int, float>>> initialize_neurons(int n, int m, float p) {
    vector<vector<pair<int, float>>> res(n, vector<pair<int, float>>());
    //Side effect: adjacency list is in order
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (dist(e2) < p) {
                res[i].emplace_back(j, 1);
            }
        }
    }
    
    return res;
}

Stimulus::Stimulus(string name, int k, float p, float default_beta) : name(name), k(k), p(p), default_beta(default_beta) {}

void Stimulus::set_beta(Area &area, float beta) {
    custom_betas[area] = beta;
}

float Stimulus::get_beta(Area &area) {
    auto it =custom_betas.find(area);
    if (it == custom_betas.end()) {
        return default_beta;
    }
    return it->second;
}

void Stimulus::project_into(Area &area) {
    if (adj.find(area) == adj.end()) {
        adj.emplace(area, initialize_neurons(k, area.n, p));
    }
    
    auto &connections = adj.find(area)->second;
    
    for (int v = 0; v < k; v++) {
        for (auto a : connections[v]) {
            int u;
            float w;
            tie(u, w) = a;
            area.activations[u] += w;
        }
    }
}

void Stimulus::update_plasticities(Area &area) {
    float beta = get_beta(area);
    auto &area_adj = adj.find(area)->second;
    //For every neuron in the stimulus
    for (int i = 0; i < k; i++) {
        auto &neighbours = area_adj[i];
        //For every winner in the area
        for (int w : area.winners) {
            //If the winner is in the adjacency list of the neuron, then increase the plasticity
            pair<int, float> a = make_pair(w, 0);
            auto it = lower_bound(neighbours.begin(), neighbours.end(), a);
            if (it != neighbours.end() && it->first == w) {
                it->second *= 1+beta;
            }
        }
    }
}

bool Stimulus::operator == (Stimulus const& other) const {
    return name == other.name;
}

bool Stimulus::operator < (Stimulus const& other) const {
    return name < other.name;
}

Area::Area(std::string name, int n, int k, float p, float default_beta) : name(name), n(n), k(k), p(p), default_beta(default_beta), activations(n) {}

void Area::set_beta(Area &area, float beta) {
    custom_betas[area] = beta;
}

float Area::get_beta(Area &area) {
    auto it = custom_betas.find(area);
    if (it == custom_betas.end()) {
        return default_beta;
    }
    return it->second;
}

void Area::project_into(Area &area) {
    //Initialise synapses to next area if they don't exist yet
    if (adj.find(area) == adj.end()) {
        adj.emplace(area, initialize_neurons(n, area.n, p));
    }
    
    auto connections = adj.find(area)->second;
    
    for (int v : winners) {
        for (auto a : connections[v]) {
            int u;
            float w;
            tie(u, w) = a;
            area.activations[u] += w;
        }
    }
}

void Area::reset_activations() {
    activations = vector<float>(n, 0);
}

void Area::recalculate_winners() {
    vector<pair<float, int>> indexed_activations;
    for (int i = 0; i < activations.size(); i++) {
        indexed_activations.emplace_back(activations[i], i);
    }
    sort(indexed_activations.begin(), indexed_activations.end(), greater<>());
    winners.clear();
    for (int i = 0; i < k; i++) {
        winners.push_back(indexed_activations[i].second);
    }
    sort(winners.begin(), winners.end());
}

void Area::save_assembly(Stimulus &stim) {
    reset_activations();
    stim.project_into(*this);
    recalculate_winners();
    saved_assemblies[stim] = winners;
}

Stimulus& Area::read_assembly() {
    pair<int, Stimulus&> best_assembly = {-1, NULL_STIM};
    for (auto &[stimulus, neurons] : saved_assemblies) {
        int common = 0;
        int j = 0;
        for (int i = 0; i < neurons.size(); i++) {
            while (j < winners.size() && winners[j] < neurons[i]) {
                j++;
            }
            if (neurons[i] == winners[j]) {
                common++;
            }
        }
        best_assembly = max(best_assembly, {common, stimulus});
    }
    return best_assembly.second;
}

void Area::update_plasticities(Area &area) {
    float beta = get_beta(area);
    
    auto &area_adj = adj.find(area)->second;
    //For every neuron that fired from the area
    for (int i : winners) {
        auto &neighbours = area_adj[i];
        //For every winner in the area
        for (int w : area.winners) {
            //If the winner is in the adjacency list of the neuron, then increase the plasticity
            pair<int, float> a = make_pair(w, 0);
            auto it = lower_bound(neighbours.begin(), neighbours.end(), a);
            if (it != neighbours.end() && it->first == w) {
                it->second *= 1+beta;
            }
        }
    }
}

bool Area::operator == (Area const& other) const {
    return name == other.name;
}

bool Area::operator < (Area const& other) const {
    return name < other.name;
}

Area& Brain::get_area(string name) {
    return *name_to_area.find(name)->second;
}

Area& Brain::add_area(std::string name, int n, int k, float p, float default_beta) {
    return *name_to_area.emplace(name, make_unique<Area>(name, n, k, p, default_beta)).first->second;
}

Stimulus& Brain::get_stimulus(std::string name) {
    return *name_to_stimulus.find(name)->second;
}

Stimulus& Brain::add_stimulus(std::string name, int k, float p, float default_beta) {
    return *name_to_stimulus.emplace(name, make_unique<Stimulus>(name, k, p, default_beta)).first->second;
}

void Brain::project(map<string, vector<string>> const &stim_to_area_str, map<string, vector<string>> const &area_to_area_str) {
    map<reference_wrapper<Stimulus>, vector<reference_wrapper<Area>>, less<Stimulus>> stim_to_area;
    map<reference_wrapper<Area>, vector<reference_wrapper<Area>>, less<Area>> area_to_area;
    for (auto &[src, dests] : stim_to_area_str) {
        vector<reference_wrapper<Area>> vec;
        for (auto &dest : dests) {
            vec.emplace_back(get_area(dest));
        }
        stim_to_area.emplace(get_stimulus(src), vec);
    }
    
    for (auto &[src, dests] : area_to_area_str) {
        vector<reference_wrapper<Area>> vec;
        for (auto &dest : dests) {
            vec.emplace_back(get_area(dest));
        }
        area_to_area.emplace(get_area(src), vec);
    }
    project(stim_to_area, area_to_area);
}

void Brain::project(map<reference_wrapper<Stimulus>, vector<reference_wrapper<Area>>, less<Stimulus>> const &stim_to_area, map<reference_wrapper<Area>, vector<reference_wrapper<Area>>, less<Area>> const &area_to_area) {
    vector<reference_wrapper<Area>> dest_name_to_area;

    for (auto &[stim, dests] : stim_to_area) {
        for (Area &a : dests) {
            dest_name_to_area.push_back(a);
        }
    }

    for (auto &[src, dests] : area_to_area) {
        for (Area &a : dests) {
            dest_name_to_area.push_back(a);
        }
    }

    sort(dest_name_to_area.begin(), dest_name_to_area.end(), less<Area>());
    
    auto it = unique(dest_name_to_area.begin(), dest_name_to_area.end(), equal_to<Area>());
    dest_name_to_area.resize(distance(dest_name_to_area.begin(), it), NULL_AREA);

    for (Area &area : dest_name_to_area) {
        area.reset_activations();
    }

    for (auto &[stim, dests] : stim_to_area) {
        for (Area &a : dests) {
            stim.get().project_into(a);
        }
    }
    for (auto &[source, dests] : area_to_area) {
        for (Area &a : dests) {
            source.get().project_into(a);
        }
    }

    for (Area &area : dest_name_to_area) {
        area.recalculate_winners();
    }

    //For every stim, check which neurons have edges to winners in each dest area
    for (auto &kvp : stim_to_area) {
        auto &[stim_wrap, dests] = kvp;
        auto &stim = stim_wrap.get();
        //For every candidate area update plasticities
        for (Area &a : dests) {
            stim.update_plasticities(a);
        }
    }
    
    //For every area, check which neurons have edges to winners in each dest area
    for (auto &kvp : area_to_area) {
        auto &[src_wrap, dests] = kvp;
        auto &src = src_wrap.get();
        //For every candidate area update plasticities
        for (Area &a : dests) {
            src.update_plasticities(a);
        }
    }
}
