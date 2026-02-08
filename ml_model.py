"""
Cell Response AI Model
Trains a neural network to predict cell responses to various chemical conditions
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json


class CellResponseAI:
    """
    AI model to predict cell responses to environmental conditions including
    unseen chemicals and combinations
    """

    def __init__(self, model_path='models/'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.chemical_database = self._initialize_chemical_database()

        # Create models directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)

        # Try to load existing model, or train new one
        if not self._load_model():
            print("Training new AI model...")
            self._train_model()
            self._save_model()
            print("Model training complete!")

    def _initialize_chemical_database(self):
        """
        Database of known chemicals and their properties
        Each chemical has: toxicity, pH_effect, oxidative_stress, membrane_disruption
        """
        return {
            # Nutrients
            "glucose": {"toxicity": 0.1, "ph_effect": 0.0, "oxidative_stress": 0.0, "membrane_disruption": 0.0},
            "amino_acids": {"toxicity": 0.05, "ph_effect": 0.1, "oxidative_stress": 0.0, "membrane_disruption": 0.0},
            "vitamins": {"toxicity": 0.0, "ph_effect": 0.0, "oxidative_stress": -0.2, "membrane_disruption": 0.0},
            "minerals": {"toxicity": 0.15, "ph_effect": 0.2, "oxidative_stress": 0.0, "membrane_disruption": 0.0},

            # Toxic chemicals
            "cyanide": {"toxicity": 0.95, "ph_effect": 0.3, "oxidative_stress": 0.8, "membrane_disruption": 0.6},
            "arsenic": {"toxicity": 0.9, "ph_effect": 0.2, "oxidative_stress": 0.7, "membrane_disruption": 0.5},
            "mercury": {"toxicity": 0.85, "ph_effect": 0.1, "oxidative_stress": 0.6, "membrane_disruption": 0.8},
            "lead": {"toxicity": 0.8, "ph_effect": 0.15, "oxidative_stress": 0.5, "membrane_disruption": 0.4},

            # Oxidative stress agents
            "hydrogen_peroxide": {"toxicity": 0.6, "ph_effect": -0.2, "oxidative_stress": 0.9,
                                  "membrane_disruption": 0.3},
            "bleach": {"toxicity": 0.75, "ph_effect": 0.8, "oxidative_stress": 0.7, "membrane_disruption": 0.5},
            "ozone": {"toxicity": 0.7, "ph_effect": 0.0, "oxidative_stress": 0.95, "membrane_disruption": 0.4},

            # pH modifiers
            "acid": {"toxicity": 0.5, "ph_effect": -0.9, "oxidative_stress": 0.2, "membrane_disruption": 0.6},
            "base": {"toxicity": 0.5, "ph_effect": 0.9, "oxidative_stress": 0.1, "membrane_disruption": 0.5},

            # Membrane disruptors
            "detergent": {"toxicity": 0.6, "ph_effect": 0.1, "oxidative_stress": 0.1, "membrane_disruption": 0.9},
            "ethanol": {"toxicity": 0.4, "ph_effect": 0.0, "oxidative_stress": 0.3, "membrane_disruption": 0.7},
            "chloroform": {"toxicity": 0.7, "ph_effect": 0.0, "oxidative_stress": 0.4, "membrane_disruption": 0.85},

            # Metabolic inhibitors
            "rotenone": {"toxicity": 0.75, "ph_effect": 0.0, "oxidative_stress": 0.8, "membrane_disruption": 0.2},
            "antimycin": {"toxicity": 0.7, "ph_effect": 0.0, "oxidative_stress": 0.75, "membrane_disruption": 0.15},

            # Protective agents
            "antioxidant": {"toxicity": -0.1, "ph_effect": 0.0, "oxidative_stress": -0.8, "membrane_disruption": 0.0},
            "buffer": {"toxicity": 0.0, "ph_effect": -0.5, "oxidative_stress": 0.0, "membrane_disruption": 0.0},

            # Common lab chemicals
            "dmso": {"toxicity": 0.3, "ph_effect": 0.0, "oxidative_stress": 0.1, "membrane_disruption": 0.4},
            "formaldehyde": {"toxicity": 0.8, "ph_effect": -0.1, "oxidative_stress": 0.6, "membrane_disruption": 0.7},
            "acetone": {"toxicity": 0.5, "ph_effect": 0.0, "oxidative_stress": 0.2, "membrane_disruption": 0.6},
        }

    def _generate_training_data(self, n_samples=10000):
        """
        Generate synthetic training data based on biological principles
        """
        X = []  # Features: [nutrients, proteins, toxicity, ph_effect, oxidative_stress, membrane_disruption]
        y = []  # Targets: [health_change, size_change, viability]

        for _ in range(n_samples):
            # Random environmental conditions
            nutrients = np.random.uniform(0, 100)
            proteins = np.random.uniform(0, 100)

            # Random chemical mixture
            n_chemicals = np.random.randint(0, 4)
            total_toxicity = 0
            total_ph = 0
            total_oxidative = 0
            total_membrane = 0

            for _ in range(n_chemicals):
                chemical_name = np.random.choice(list(self.chemical_database.keys()))
                concentration = np.random.uniform(0, 100)
                props = self.chemical_database[chemical_name]

                total_toxicity += props["toxicity"] * concentration / 100
                total_ph += props["ph_effect"] * concentration / 100
                total_oxidative += props["oxidative_stress"] * concentration / 100
                total_membrane += props["membrane_disruption"] * concentration / 100

            # Normalize
            total_toxicity = min(100, total_toxicity * 20)
            total_ph = np.clip(total_ph * 20, -100, 100)
            total_oxidative = min(100, total_oxidative * 20)
            total_membrane = min(100, total_membrane * 20)

            features = [nutrients, proteins, total_toxicity, total_ph,
                        total_oxidative, total_membrane]

            # Calculate biological responses
            health_change = self._calculate_health_change(
                nutrients, proteins, total_toxicity, total_ph,
                total_oxidative, total_membrane
            )

            size_change = self._calculate_size_change(
                nutrients, proteins, health_change
            )

            viability = self._calculate_viability(
                health_change, total_toxicity, total_membrane
            )

            X.append(features)
            y.append([health_change, size_change, viability])

        return np.array(X), np.array(y)

    def _calculate_health_change(self, nutrients, proteins, toxicity,
                                 ph_effect, oxidative, membrane):
        """Calculate health change based on multiple factors"""
        health_change = 0

        # Nutrient effects (optimal: 40-70)
        if 40 <= nutrients <= 70:
            health_change += 8
        elif nutrients < 20 or nutrients > 90:
            health_change -= 15
        else:
            health_change -= 5

        # Protein effects (optimal: 30-60)
        if 30 <= proteins <= 60:
            health_change += 7
        elif proteins < 15 or proteins > 85:
            health_change -= 12
        else:
            health_change -= 4

        # Toxicity damage
        health_change -= toxicity * 0.4

        # pH stress (optimal near 0)
        ph_stress = abs(ph_effect) * 0.3
        health_change -= ph_stress

        # Oxidative stress
        health_change -= oxidative * 0.35

        # Membrane damage
        health_change -= membrane * 0.5

        # Add some biological noise
        health_change += np.random.normal(0, 2)

        return np.clip(health_change, -50, 30)

    def _calculate_size_change(self, nutrients, proteins, health_change):
        """Calculate size change based on growth conditions"""
        if health_change > 5 and nutrients > 50 and proteins > 40:
            return np.random.uniform(0.01, 0.05)
        elif health_change < -10:
            return np.random.uniform(-0.05, -0.02)
        else:
            return np.random.uniform(-0.01, 0.01)

    def _calculate_viability(self, health_change, toxicity, membrane):
        """Calculate cell viability (0-1)"""
        base_viability = 0.8

        if health_change > 0:
            base_viability += 0.1
        else:
            base_viability += health_change * 0.01

        # Severe toxicity reduces viability
        base_viability -= (toxicity / 100) * 0.6

        # Membrane damage is critical
        base_viability -= (membrane / 100) * 0.5

        return np.clip(base_viability, 0, 1)

    def _train_model(self):
        """Train the neural network model"""
        print("Generating training data...")
        X_train, y_train = self._generate_training_data(n_samples=15000)

        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train neural network
        print("Training neural network...")
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        )

        self.model.fit(X_train_scaled, y_train)

        # Calculate training score
        score = self.model.score(X_train_scaled, y_train)
        print(f"Model R² score: {score:.4f}")

    def _save_model(self):
        """Save trained model and scaler"""
        joblib.dump(self.model, os.path.join(self.model_path, 'cell_model.pkl'))
        joblib.dump(self.scaler, os.path.join(self.model_path, 'scaler.pkl'))

        # Save chemical database
        with open(os.path.join(self.model_path, 'chemical_db.json'), 'w') as f:
            json.dump(self.chemical_database, f, indent=2)

    def _load_model(self):
        """Load existing model and scaler"""
        try:
            self.model = joblib.load(os.path.join(self.model_path, 'cell_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_path, 'scaler.pkl'))

            # Load chemical database
            db_path = os.path.join(self.model_path, 'chemical_db.json')
            if os.path.exists(db_path):
                with open(db_path, 'r') as f:
                    self.chemical_database = json.load(f)

            print("Loaded existing AI model")
            return True
        except:
            return False

    def predict_response(self, nutrients, proteins, chemicals_list):
        """
        Predict cell response to given conditions

        Parameters:
        - nutrients: 0-100
        - proteins: 0-100
        - chemicals_list: list of dicts with 'name' and 'concentration'
          e.g., [{"name": "cyanide", "concentration": 50}, {"name": "glucose", "concentration": 30}]

        Returns:
        - dict with health_change, size_change, viability, and analysis
        """
        # Calculate chemical effects
        total_toxicity = 0
        total_ph = 0
        total_oxidative = 0
        total_membrane = 0
        unknown_chemicals = []

        for chem in chemicals_list:
            name = chem['name'].lower()
            concentration = chem['concentration']

            if name in self.chemical_database:
                props = self.chemical_database[name]
                total_toxicity += props["toxicity"] * concentration / 100
                total_ph += props["ph_effect"] * concentration / 100
                total_oxidative += props["oxidative_stress"] * concentration / 100
                total_membrane += props["membrane_disruption"] * concentration / 100
            else:
                # Unknown chemical - estimate based on name patterns and add randomness
                unknown_chemicals.append(name)
                # Use heuristics for unknown chemicals
                estimated_props = self._estimate_chemical_properties(name)
                total_toxicity += estimated_props["toxicity"] * concentration / 100
                total_ph += estimated_props["ph_effect"] * concentration / 100
                total_oxidative += estimated_props["oxidative_stress"] * concentration / 100
                total_membrane += estimated_props["membrane_disruption"] * concentration / 100

        # Normalize
        total_toxicity = min(100, total_toxicity * 20)
        total_ph = np.clip(total_ph * 20, -100, 100)
        total_oxidative = min(100, total_oxidative * 20)
        total_membrane = min(100, total_membrane * 20)

        # Create feature vector
        features = np.array([[nutrients, proteins, total_toxicity, total_ph,
                              total_oxidative, total_membrane]])

        # Scale and predict
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)[0]

        health_change, size_change, viability = predictions

        # Generate analysis
        analysis = self._generate_analysis(
            nutrients, proteins, total_toxicity, total_ph,
            total_oxidative, total_membrane, health_change,
            viability, unknown_chemicals
        )

        return {
            "health_change": float(health_change),
            "size_change": float(size_change),
            "viability": float(viability),
            "chemical_effects": {
                "toxicity": float(total_toxicity),
                "ph_effect": float(total_ph),
                "oxidative_stress": float(total_oxidative),
                "membrane_disruption": float(total_membrane)
            },
            "analysis": analysis,
            "unknown_chemicals": unknown_chemicals
        }

    def _estimate_chemical_properties(self, name):
        """Estimate properties for unknown chemicals based on name patterns"""
        name = name.lower()

        # Default moderate toxicity
        props = {
            "toxicity": 0.5,
            "ph_effect": 0.0,
            "oxidative_stress": 0.3,
            "membrane_disruption": 0.3
        }

        # Heuristics based on common chemical naming
        if any(word in name for word in ['acid', 'acidic']):
            props["ph_effect"] = -0.7
            props["toxicity"] = 0.6

        if any(word in name for word in ['base', 'basic', 'alkaline']):
            props["ph_effect"] = 0.7
            props["toxicity"] = 0.6

        if any(word in name for word in ['toxic', 'poison', 'venom']):
            props["toxicity"] = 0.85
            props["oxidative_stress"] = 0.6

        if any(word in name for word in ['peroxide', 'oxide', 'oxygen']):
            props["oxidative_stress"] = 0.8
            props["toxicity"] = 0.5

        if any(word in name for word in ['detergent', 'soap', 'solvent']):
            props["membrane_disruption"] = 0.8
            props["toxicity"] = 0.5

        if any(word in name for word in ['nutrient', 'vitamin', 'food']):
            props["toxicity"] = 0.1
            props["oxidative_stress"] = -0.2

        # Add randomness for truly unknown chemicals
        for key in props:
            props[key] += np.random.normal(0, 0.1)
            props[key] = np.clip(props[key], -1, 1)

        return props

    def _generate_analysis(self, nutrients, proteins, toxicity, ph,
                           oxidative, membrane, health_change, viability,
                           unknown_chemicals):
        """Generate human-readable analysis"""
        analysis = []

        # Nutrient analysis
        if nutrients < 30:
            analysis.append("Severe nutrient deficiency detected")
        elif nutrients < 40:
            analysis.append("Low nutrient levels")
        elif nutrients > 80:
            analysis.append("Excess nutrients causing osmotic stress")

        # Protein analysis
        if proteins < 25:
            analysis.append("Critical protein shortage")
        elif proteins > 70:
            analysis.append("Protein aggregation risk")

        # Chemical stress analysis
        if toxicity > 60:
            analysis.append("SEVERE TOXICITY - immediate cell damage")
        elif toxicity > 30:
            analysis.append("Moderate toxic exposure")

        if abs(ph) > 50:
            analysis.append("CRITICAL pH imbalance")
        elif abs(ph) > 25:
            analysis.append("pH stress detected")

        if oxidative > 50:
            analysis.append("Severe oxidative damage")
        elif oxidative > 25:
            analysis.append("Oxidative stress present")

        if membrane > 60:
            analysis.append("CRITICAL membrane disruption")
        elif membrane > 30:
            analysis.append("Membrane integrity compromised")

        # Overall prediction
        if health_change > 5:
            analysis.append("Cells predicted to thrive")
        elif health_change > 0:
            analysis.append("Stable conditions")
        elif health_change > -10:
            analysis.append("Cells will deteriorate slowly")
        else:
            analysis.append("Rapid cell death expected")

        if viability < 0.3:
            analysis.append("Cell viability critically low")
        elif viability < 0.6:
            analysis.append("Reduced cell viability")

        # Unknown chemicals warning
        if unknown_chemicals:
            analysis.append(f"Unknown chemicals detected: {', '.join(unknown_chemicals[:3])}")
            analysis.append("AI model estimated their effects")

        return analysis

    def get_chemical_list(self):
        """Return list of known chemicals"""
        return sorted(list(self.chemical_database.keys()))

    def get_chemical_info(self, chemical_name):
        """Get information about a specific chemical"""
        chemical_name = chemical_name.lower()
        if chemical_name in self.chemical_database:
            return self.chemical_database[chemical_name]
        else:
            return self._estimate_chemical_properties(chemical_name)
