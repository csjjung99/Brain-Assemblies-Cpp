//
//  brain.hpp
//  Brain
//
//  Created by Chris Jung on 9/09/21.
//  Copyright © 2021 Chris Jung. All rights reserved.
//

#ifndef brain_hpp
#define brain_hpp

#include <map>
#include <string>
#include <vector>
#include <memory>


class Area;
class Brain;

class Stimulus {
    std::map<std::reference_wrapper<Area>, std::vector<std::vector<std::pair<int, double>>>, std::less<Area>> adj;
    std::map<std::reference_wrapper<Area>, double, std::less<Area>> custom_betas;
public:
    std::string name;
    
    int k;
    double p;
    double default_beta;
    
    Stimulus(std::string name, int k, double p, double default_beta);
    Stimulus(const Stimulus& s) = delete;
    
    void set_beta(Area &area, double beta);
    double get_beta(Area &area);
    void project_into(Area &area);
    void update_plasticities(Area &area);
    bool operator == (Stimulus const& other) const;
    bool operator < (Stimulus const& other) const;
};

extern Stimulus NULL_STIM;

class Area {
    //Adjacency list, always sorted
    std::map<std::reference_wrapper<Area>, std::vector<std::vector<std::pair<int, double>>>, std::less<Area>> adj;
    //The k neurons that activated when the stimulus was fired on its own
    std::map<std::reference_wrapper<Stimulus>, std::vector<int>, std::less<Stimulus>> saved_assemblies;
    std::map<std::reference_wrapper<Area>, double, std::less<Area>> custom_betas;
public:
    //List of winners, always sorted
    std::vector<int> winners;
    std::string name;
    int n;
    int k;
    double p;
    double default_beta;
    std::vector<double> activations;
    
    Area(std::string name, int n, int k, double p, double default_beta);
    Area(const Area& a) = delete;
    
    void set_beta(Area &area, double beta);
    double get_beta(Area &area);
    void project_into(Area &area);
    void reset_activations();
    void recalculate_winners();
    void update_plasticities(Area &area);
    void save_assembly(Stimulus &stim);
    Stimulus& read_assembly();
    bool operator == (Area const& other) const;
    bool operator < (Area const& other) const;
};

extern Area NULL_AREA;

class Brain {
    std::map<std::string, std::unique_ptr<Area>> name_to_area;
    std::map<std::string, std::unique_ptr<Stimulus>> name_to_stimulus;
public:
    Area& get_area(std::string name);
    Area& add_area(std::string name, int n, int k, double p, double default_beta);
    
    Stimulus& get_stimulus(std::string name);
    Stimulus& add_stimulus(std::string name, int k, double p, double default_beta);

    void project(std::map<std::string, std::vector<std::string>> const &stim_to_area_str, std::map<std::string, std::vector<std::string>> const &area_to_area_str);
    void project(std::map<std::reference_wrapper<Stimulus>, std::vector<std::reference_wrapper<Area>>, std::less<Stimulus>> const &stim_to_area_str, std::map<std::reference_wrapper<Area>, std::vector<std::reference_wrapper<Area>>, std::less<Area>> const &area_to_area_str);
};


#endif /* brain_hpp */
