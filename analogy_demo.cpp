//
//  main.cpp
//  Brain
//
//  Created by Chris Jung on 17/09/21.
//  Copyright © 2021 Chris Jung. All rights reserved.
//

#include <iostream>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <fstream>
#include <random>
using namespace std;

#include "brain.hpp"

float p = 0.01;
float beta = 0.05;
int area_size = 2500;
int assembly_size = 50;

Brain b;
Area &vocab_area_1 = b.add_area("var1", area_size, assembly_size, p, beta);
Area &vocab_area_2 = b.add_area("var2", area_size, assembly_size, p, beta);
Area &word_pair_area = b.add_area("wpa", area_size, assembly_size, p, beta);
Area &relation_area = b.add_area("ra", area_size, assembly_size, p, beta);
Area &output_area = b.add_area("oa", area_size, assembly_size, p, beta);

static random_device rd;
static mt19937 e2(rd());

string solve(string A, string B, string C) {
    Stimulus &stimA = b.get_stimulus("stim" + A);
    Stimulus &stimB = b.get_stimulus("stim" + B);
    Stimulus &stimC = b.get_stimulus("stim" + C);
    
    b.project({{stimA, {vocab_area_1}}, {stimB, {vocab_area_2}}}, {});
    
    b.project({}, {{vocab_area_1, {word_pair_area}}, {vocab_area_2, {word_pair_area}}});

    b.project({}, {{word_pair_area, {relation_area}}});

    b.project({{stimC, {vocab_area_1}}}, {});

    b.project({}, {{relation_area, {output_area}}, {vocab_area_1, {output_area}}});
    Stimulus &s = output_area.read_assembly();
    
    return s.name;
}

int main() {
    set<string> vocab, relations;
    set<pair<string, string>> word_pairs;
    vector<tuple<string, string, string>> input_relations;
    {
        cout << "Parsing Input..." << endl;
        string inp;
        ifstream stream;
        stream.open("relations.csv");
        if (!stream.is_open()) {
            cout << "Error reading file. Please place 'relations.csv' in the same folder as the executable." << endl;
            return 0;
        }
        while (getline(stream, inp)) {
            string w1;
            string w2;
            string r;
            int i = 0;
            while (inp[i] != ',') {
                w1.push_back(inp[i]);
                i++;
            }
            i++;
            while (inp[i] != ',') {
                w2.push_back(inp[i]);
                i++;
            }
            i++;
            while (i < inp.size()) {
                r.push_back(inp[i]);
                i++;
            }
            input_relations.emplace_back(w1, w2, r);
            vocab.insert(w1);
            vocab.insert(w2);
            relations.insert(r);
            word_pairs.emplace(w1, w2);
        }
    }
    
    cout << "Training..." << endl;
    
    for (auto word : vocab) {
        b.add_stimulus("stim" + word, assembly_size, p, beta);
    }
    for (auto line : input_relations) {
        auto &[w1, w2, r] = line;
        b.add_stimulus("stim" + w1 + w2, assembly_size, p, beta);
        b.add_stimulus("stim" + r, assembly_size, p, beta);
    }
    for (auto line : input_relations) {
        auto &[w1, w2, r] = line;
        Stimulus &stimW1 = b.get_stimulus("stim" + w1);
        Stimulus &stimW2 = b.get_stimulus("stim" + w2);
        Stimulus &stimW1W2 = b.get_stimulus("stim" + w1 + w2);
        Stimulus &stimR = b.get_stimulus("stim" + r);
        
        for (int i = 0; i < 100; i++) {
            b.project({{stimW1, {vocab_area_1, vocab_area_2}}}, {});
        }
        
        for (int i = 0; i < 100; i++) {
            b.project({{stimW2, {vocab_area_1, vocab_area_2}}}, {});
        }

        b.project({{stimW1, {vocab_area_1}}, {stimW2, {vocab_area_2}}}, {});
        for (int i = 0; i < 100; i++) {
            b.project({{stimW1W2, {word_pair_area}}}, {{vocab_area_1, {word_pair_area}}, {vocab_area_2, {word_pair_area}}});
        }

        b.project({{stimW1, {vocab_area_2}}, {stimW2, {vocab_area_1}}}, {});
        for (int i = 0; i < 100; i++) {
            b.project({{stimW1W2, {word_pair_area}}}, {{vocab_area_1, {word_pair_area}}, {vocab_area_2, {word_pair_area}}});
        }

        //Project Relation + Word pair into the relation area
        for (int i = 0; i < 100; i++) {
            b.project({{stimR, {relation_area}}}, {{word_pair_area, {relation_area}}});
        }

        b.project({{stimW1, {vocab_area_1}}}, {});
        for (int i = 0; i < 100; i++) {
            b.project({{stimW1W2, {output_area}}}, {{vocab_area_1, {output_area}}});
        }
        b.project({{stimW2, {vocab_area_1}}}, {});
        for (int i = 0; i < 100; i++) {
            b.project({{stimW1W2, {output_area}}}, {{vocab_area_1, {output_area}}});
        }
    }
    shuffle(input_relations.begin(), input_relations.end(), e2);
    double reduced_beta = 0.001;
    vocab_area_1.default_beta = reduced_beta;
    relation_area.default_beta = reduced_beta;
    word_pair_area.default_beta = reduced_beta;
    for (int i = 0; i < 300; i++) {
        for (auto line : input_relations) {
            auto &[w1, w2, r] = line;
            Stimulus &stimW1 = b.get_stimulus("stim" + w1);
            Stimulus &stimW2 = b.get_stimulus("stim" + w2);
            Stimulus &stimW1W2 = b.get_stimulus("stim" + w1 + w2);
            Stimulus &stimR = b.get_stimulus("stim" + r);
            stimW1.default_beta = reduced_beta;
            stimW2.default_beta = reduced_beta;
            stimW1W2.default_beta = reduced_beta;
            stimR.default_beta = reduced_beta;
            b.project({{stimW1, {vocab_area_1}}, {stimW1W2, {word_pair_area}}, {stimR, {relation_area}}}, {});
            b.project({}, {{vocab_area_1, {output_area}}, {word_pair_area, {output_area}}, {relation_area, {output_area}}});
        }
    }
    
    for (auto word : vocab) {
        vocab_area_1.save_assembly(b.get_stimulus("stim" + word));
        vocab_area_2.save_assembly(b.get_stimulus("stim" + word));
    }

    for (auto word_pair : word_pairs) {
        word_pair_area.save_assembly(b.get_stimulus("stim" + word_pair.first + word_pair.second));
        output_area.save_assembly(b.get_stimulus("stim" + word_pair.first + word_pair.second));
    }

    for (auto relation : relations) {
        relation_area.save_assembly(b.get_stimulus("stim" + relation));
    }
    
    cout << "Ready For Input" << endl;
    
    while (true) {
        cout << "Enter three space separated words" << endl;
        string A, B, C;
        cin >> A >> B >> C;
        cout << "Solving " << A << ":" << B << "::" << C << ":?" << endl;
        cout << "Found Solution: " << solve(A, B, C) << endl << endl;
    }
}
