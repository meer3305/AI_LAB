"""
Expert System with Forward Chaining Implementation
Author: AI Lab
Description: Simple expert system using forward chaining inference
"""


class Rule:
    def __init__(self, conditions, conclusion, name=None):
        """
        conditions: list of facts that must be true
        conclusion: fact that will be derived if conditions are met
        name: optional name for the rule
        """
        self.conditions = conditions
        self.conclusion = conclusion
        self.name = name or f"Rule: {' AND '.join(conditions)} -> {conclusion}"
    
    def can_fire(self, facts):
        """Check if all conditions are satisfied by current facts"""
        return all(condition in facts for condition in self.conditions)
    
    def fire(self, facts):
        """Apply the rule and return new fact if conditions are met"""
        if self.can_fire(facts):
            return self.conclusion
        return None


class ExpertSystem:
    def __init__(self):
        self.rules = []
        self.facts = set()
        self.fired_rules = []
    
    def add_rule(self, rule):
        """Add a rule to the expert system"""
        self.rules.append(rule)
    
    def add_fact(self, fact):
        """Add a fact to the knowledge base"""
        self.facts.add(fact)
    
    def forward_chain(self, verbose=True):
        """Perform forward chaining inference"""
        if verbose:
            print("=== Forward Chaining Inference ===")
            print(f"Initial facts: {sorted(self.facts)}")
        
        iteration = 1
        new_facts_added = True
        
        while new_facts_added:
            new_facts_added = False
            
            if verbose:
                print(f"\nIteration {iteration}:")
            
            for rule in self.rules:
                if rule not in self.fired_rules and rule.can_fire(self.facts):
                    new_fact = rule.fire(self.facts)
                    
                    if new_fact and new_fact not in self.facts:
                        self.facts.add(new_fact)
                        self.fired_rules.append(rule)
                        new_facts_added = True
                        
                        if verbose:
                            print(f"  Fired: {rule.name}")
                            print(f"  New fact: {new_fact}")
            
            if verbose and not new_facts_added:
                print("  No new facts derived")
            
            iteration += 1
        
        if verbose:
            print(f"\nFinal facts: {sorted(self.facts)}")
            print(f"Total rules fired: {len(self.fired_rules)}")
        
        return self.facts


def create_animal_classification_system():
    """Create an expert system for animal classification"""
    expert_system = ExpertSystem()
    
    # Add rules for animal classification
    rules = [
        Rule(["has_hair"], "is_mammal", "R1: Has hair -> Mammal"),
        Rule(["gives_milk"], "is_mammal", "R2: Gives milk -> Mammal"),
        Rule(["has_feathers"], "is_bird", "R3: Has feathers -> Bird"),
        Rule(["flies", "lays_eggs"], "is_bird", "R4: Flies and lays eggs -> Bird"),
        Rule(["is_mammal", "eats_meat"], "is_carnivore", "R5: Mammal that eats meat -> Carnivore"),
        Rule(["is_mammal", "has_pointed_teeth", "has_claws", "forward_eyes"], "is_carnivore", 
              "R6: Mammal with hunting features -> Carnivore"),
        Rule(["is_mammal", "has_hoofs"], "is_ungulate", "R7: Mammal with hoofs -> Ungulate"),
        Rule(["is_mammal", "chews_cud"], "is_ungulate", "R8: Mammal that chews cud -> Ungulate"),
        Rule(["is_carnivore", "has_tawny_color", "has_dark_spots"], "is_cheetah", 
              "R9: Carnivore with tawny color and dark spots -> Cheetah"),
        Rule(["is_carnivore", "has_tawny_color", "has_black_stripes"], "is_tiger", 
              "R10: Carnivore with tawny color and black stripes -> Tiger"),
        Rule(["is_ungulate", "has_long_neck", "has_long_legs", "has_dark_spots"], "is_giraffe", 
              "R11: Ungulate with long neck, legs, and dark spots -> Giraffe"),
        Rule(["is_ungulate", "has_black_stripes"], "is_zebra", 
              "R12: Ungulate with black stripes -> Zebra"),
        Rule(["is_bird", "does_not_fly", "has_long_neck", "has_long_legs", "is_black_and_white"], "is_ostrich", 
              "R13: Flightless bird with long neck/legs, black and white -> Ostrich"),
        Rule(["is_bird", "swims", "is_black_and_white"], "is_penguin", 
              "R14: Bird that swims, black and white -> Penguin"),
    ]
    
    for rule in rules:
        expert_system.add_rule(rule)
    
    return expert_system


def demonstrate_expert_system():
    """Demonstrate expert system with animal classification"""
    print("=== Expert System Demonstration ===")
    print("Animal Classification Expert System")
    
    # Test case 1: Cheetah
    print("\n" + "="*50)
    print("TEST CASE 1: Identifying a Cheetah")
    print("="*50)
    
    es1 = create_animal_classification_system()
    
    # Add facts about the animal
    cheetah_facts = [
        "has_hair",
        "eats_meat", 
        "has_pointed_teeth",
        "has_claws",
        "forward_eyes",
        "has_tawny_color",
        "has_dark_spots"
    ]
    
    for fact in cheetah_facts:
        es1.add_fact(fact)
    
    # Perform inference
    final_facts = es1.forward_chain()
    
    # Test case 2: Giraffe
    print("\n" + "="*50)
    print("TEST CASE 2: Identifying a Giraffe")
    print("="*50)
    
    es2 = create_animal_classification_system()
    
    giraffe_facts = [
        "has_hair",
        "has_hoofs",
        "has_long_neck",
        "has_long_legs",
        "has_dark_spots"
    ]
    
    for fact in giraffe_facts:
        es2.add_fact(fact)
    
    final_facts = es2.forward_chain()
    
    # Test case 3: Penguin
    print("\n" + "="*50)
    print("TEST CASE 3: Identifying a Penguin")
    print("="*50)
    
    es3 = create_animal_classification_system()
    
    penguin_facts = [
        "has_feathers",
        "swims",
        "is_black_and_white"
    ]
    
    for fact in penguin_facts:
        es3.add_fact(fact)
    
    final_facts = es3.forward_chain()


def create_medical_diagnosis_system():
    """Create a simple medical diagnosis expert system"""
    expert_system = ExpertSystem()
    
    # Medical diagnosis rules
    rules = [
        Rule(["fever", "headache", "body_aches"], "flu_symptoms", "R1: Fever, headache, body aches -> Flu symptoms"),
        Rule(["flu_symptoms", "season_winter"], "likely_flu", "R2: Flu symptoms in winter -> Likely flu"),
        Rule(["cough", "chest_pain", "difficulty_breathing"], "respiratory_symptoms", "R3: Respiratory issues"),
        Rule(["respiratory_symptoms", "fever"], "possible_pneumonia", "R4: Respiratory symptoms with fever -> Possible pneumonia"),
        Rule(["sore_throat", "fever", "swollen_glands"], "throat_infection", "R5: Throat infection symptoms"),
        Rule(["throat_infection", "white_patches"], "strep_throat", "R6: Throat infection with white patches -> Strep throat"),
        Rule(["runny_nose", "sneezing", "mild_cough"], "cold_symptoms", "R7: Cold symptoms"),
        Rule(["cold_symptoms", "no_fever"], "common_cold", "R8: Cold symptoms without fever -> Common cold"),
    ]
    
    for rule in rules:
        expert_system.add_rule(rule)
    
    return expert_system


def demonstrate_medical_diagnosis():
    """Demonstrate medical diagnosis expert system"""
    print("\n" + "="*60)
    print("MEDICAL DIAGNOSIS EXPERT SYSTEM")
    print("="*60)
    
    es = create_medical_diagnosis_system()
    
    # Patient symptoms
    patient_symptoms = [
        "fever",
        "headache", 
        "body_aches",
        "season_winter"
    ]
    
    print(f"Patient symptoms: {patient_symptoms}")
    
    for symptom in patient_symptoms:
        es.add_fact(symptom)
    
    final_facts = es.forward_chain()
    
    # Extract diagnoses
    diagnoses = [fact for fact in final_facts if fact.startswith(('likely_', 'possible_', 'common_', 'strep_'))]
    if diagnoses:
        print(f"\nPossible diagnoses: {diagnoses}")
    else:
        print("\nNo specific diagnosis could be determined.")


if __name__ == "__main__":
    demonstrate_expert_system()
    demonstrate_medical_diagnosis()