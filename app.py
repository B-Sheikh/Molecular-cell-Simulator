"""
Interactive 3D Cell Simulator - Flask Backend
This server handles cell simulation calculations and serves the web interface
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import json
from ml_model import CellResponseAI

app = Flask(__name__)

# Initialize AI model
print("Initializing AI model...")
ai_model = CellResponseAI()

@dataclass
class CellState:
    """Represents the current state of a cell"""
    health: float = 100.0
    size: float = 1.0
    color: str = "#00ff00"
    nutrients: float = 50.0
    proteins: float = 50.0
    toxicity: float = 0.0

class CellSimulator:
    """Simulates cell behavior based on environmental conditions"""

    def __init__(self):
        self.cells = {}
        self.initialize_cells()

    def initialize_cells(self):
        """Create initial set of cells"""
        for i in range(3):
            self.cells[f"cell_{i}"] = CellState(
                health=100.0,
                size=1.0 + np.random.uniform(-0.2, 0.2),
                nutrients=50.0,
                proteins=50.0,
                toxicity=0.0
            )

    def calculate_cell_response_ai(self, cell_id: str, nutrients: float,
                                   proteins: float, chemicals: List[Dict]) -> Dict:
        """
        Calculate cell response using AI model for complex chemical interactions

        Parameters:
        - nutrients: 0-100
        - proteins: 0-100
        - chemicals: list of {"name": str, "concentration": float}
        """
        cell = self.cells.get(cell_id, CellState())

        # Use AI model to predict response
        prediction = ai_model.predict_response(nutrients, proteins, chemicals)

        # Update cell state based on AI predictions
        health_change = prediction['health_change']
        size_change = prediction['size_change']
        viability = prediction['viability']

        # Apply health change
        cell.health = max(0, min(100, cell.health + health_change))

        # Apply size change
        cell.size = max(0.3, min(2.0, cell.size + size_change))

        # Update nutrients and proteins
        cell.nutrients = nutrients
        cell.proteins = proteins

        # Calculate effective toxicity from chemical effects
        chem_effects = prediction['chemical_effects']
        cell.toxicity = (chem_effects['toxicity'] +
                        abs(chem_effects['ph_effect']) * 0.5 +
                        chem_effects['oxidative_stress'] * 0.8 +
                        chem_effects['membrane_disruption']) / 3

        # Color based on health and viability
        if cell.health > 70 and viability > 0.7:
            cell.color = "#00ff00"  # Green - healthy
        elif cell.health > 50 or viability > 0.5:
            cell.color = "#ffff00"  # Yellow - stressed
        elif cell.health > 30 or viability > 0.3:
            cell.color = "#ff8800"  # Orange - struggling
        else:
            cell.color = "#ff0000"  # Red - dying

        self.cells[cell_id] = cell

        return {
            "cell_id": cell_id,
            "health": round(cell.health, 2),
            "size": round(cell.size, 2),
            "color": cell.color,
            "nutrients": round(cell.nutrients, 2),
            "proteins": round(cell.proteins, 2),
            "toxicity": round(cell.toxicity, 2),
            "viability": round(viability * 100, 2),
            "status": self._get_ai_status_message(cell, viability),
            "chemical_effects": prediction['chemical_effects'],
            "analysis": prediction['analysis'],
            "unknown_chemicals": prediction.get('unknown_chemicals', [])
        }

    def _get_ai_status_message(self, cell: CellState, viability: float) -> str:
        """Generate status message based on cell condition and viability"""
        if cell.health > 80 and viability > 0.8:
            return "Thriving - optimal conditions"
        elif cell.health > 60 and viability > 0.6:
            return "Healthy - good conditions"
        elif cell.health > 40 and viability > 0.4:
            return "Stressed - suboptimal conditions"
        elif cell.health > 20 or viability > 0.2:
            return "Struggling - poor conditions"
        else:
            return "Critical - cell death imminent"

    def update_all_cells_ai(self, nutrients: float, proteins: float,
                           chemicals: List[Dict]) -> List[Dict]:
        """Update all cells with AI predictions"""
        results = []
        for cell_id in self.cells.keys():
            result = self.calculate_cell_response_ai(
                cell_id, nutrients, proteins, chemicals
            )
            results.append(result)
        return results

    def calculate_cell_response(self, cell_id: str, nutrients: float,
                               proteins: float, chemicals: float) -> Dict:
        """
        Calculate how a cell responds to environmental conditions

        Parameters:
        - nutrients: 0-100 (optimal: 40-70)
        - proteins: 0-100 (optimal: 30-60)
        - chemicals: 0-100 (higher = more toxic)
        """
        cell = self.cells.get(cell_id, CellState())

        # Update nutrient and protein levels
        cell.nutrients = nutrients
        cell.proteins = proteins
        cell.toxicity = chemicals

        # Calculate health based on conditions
        health_change = 0

        # Nutrient effects
        if 40 <= nutrients <= 70:
            health_change += 5  # Optimal range
        elif nutrients < 20 or nutrients > 90:
            health_change -= 10  # Critical levels
        else:
            health_change -= 3  # Suboptimal

        # Protein effects
        if 30 <= proteins <= 60:
            health_change += 5  # Optimal range
        elif proteins < 15 or proteins > 85:
            health_change -= 8
        else:
            health_change -= 2

        # Chemical toxicity
        toxicity_damage = -chemicals * 0.3
        health_change += toxicity_damage

        # Update health (bounded 0-100)
        cell.health = max(0, min(100, cell.health + health_change))

        # Size changes based on nutrients and proteins
        if cell.health > 70 and nutrients > 50 and proteins > 40:
            cell.size = min(1.5, cell.size + 0.02)
        elif cell.health < 30:
            cell.size = max(0.5, cell.size - 0.03)

        # Color based on health
        if cell.health > 70:
            cell.color = "#00ff00"  # Green - healthy
        elif cell.health > 40:
            cell.color = "#ffff00"  # Yellow - stressed
        elif cell.health > 20:
            cell.color = "#ff8800"  # Orange - struggling
        else:
            cell.color = "#ff0000"  # Red - dying

        self.cells[cell_id] = cell

        return {
            "cell_id": cell_id,
            "health": round(cell.health, 2),
            "size": round(cell.size, 2),
            "color": cell.color,
            "nutrients": round(cell.nutrients, 2),
            "proteins": round(cell.proteins, 2),
            "toxicity": round(cell.toxicity, 2),
            "status": self._get_status_message(cell)
        }

    def _get_status_message(self, cell: CellState) -> str:
        """Generate status message based on cell condition"""
        if cell.health > 80:
            return "Thriving - optimal conditions"
        elif cell.health > 60:
            return "Healthy - good conditions"
        elif cell.health > 40:
            return "Stressed - suboptimal conditions"
        elif cell.health > 20:
            return "Struggling - poor conditions"
        else:
            return "Critical - cell dying"

    def update_all_cells(self, nutrients: float, proteins: float,
                        chemicals: float) -> List[Dict]:
        """Update all cells with the same environmental conditions"""
        results = []
        for cell_id in self.cells.keys():
            result = self.calculate_cell_response(
                cell_id, nutrients, proteins, chemicals
            )
            results.append(result)
        return results

    def reset_simulation(self):
        """Reset all cells to initial state"""
        self.initialize_cells()

# Global simulator instance
simulator = CellSimulator()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Handle simulation requests"""
    try:
        data = request.json
        nutrients = float(data.get('nutrients', 50))
        proteins = float(data.get('proteins', 50))
        chemicals = float(data.get('chemicals', 0))

        # Validate inputs
        nutrients = max(0, min(100, nutrients))
        proteins = max(0, min(100, proteins))
        chemicals = max(0, min(100, chemicals))

        results = simulator.update_all_cells(nutrients, proteins, chemicals)

        return jsonify({
            "success": True,
            "cells": results
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route('/api/simulate-ai', methods=['POST'])
def simulate_ai():
    """Handle AI-powered simulation with custom chemicals"""
    try:
        data = request.json
        nutrients = float(data.get('nutrients', 50))
        proteins = float(data.get('proteins', 50))
        chemicals = data.get('chemicals', [])  # List of {"name": str, "concentration": float}

        # Validate inputs
        nutrients = max(0, min(100, nutrients))
        proteins = max(0, min(100, proteins))

        # Validate chemicals list
        if not isinstance(chemicals, list):
            chemicals = []

        for chem in chemicals:
            if 'concentration' in chem:
                chem['concentration'] = max(0, min(100, float(chem['concentration'])))

        results = simulator.update_all_cells_ai(nutrients, proteins, chemicals)

        return jsonify({
            "success": True,
            "cells": results,
            "ai_powered": True
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route('/api/chemicals', methods=['GET'])
def get_chemicals():
    """Get list of known chemicals"""
    chemicals = ai_model.get_chemical_list()
    return jsonify({
        "success": True,
        "chemicals": chemicals,
        "count": len(chemicals)
    })

@app.route('/api/chemical-info/<chemical_name>', methods=['GET'])
def get_chemical_info(chemical_name):
    """Get information about a specific chemical"""
    info = ai_model.get_chemical_info(chemical_name)
    return jsonify({
        "success": True,
        "chemical": chemical_name,
        "properties": info
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the simulation"""
    simulator.reset_simulation()
    return jsonify({
        "success": True,
        "message": "Simulation reset"
    })

@app.route('/api/cell-info')
def cell_info():
    """Get information about optimal conditions"""
    return jsonify({
        "optimal_ranges": {
            "nutrients": "40-70",
            "proteins": "30-60",
            "chemicals": "0-10 (lower is better)"
        },
        "critical_levels": {
            "nutrients": "<20 or >90",
            "proteins": "<15 or >85",
            "chemicals": ">50"
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("3D Cell Simulator Server Starting...")
    print("="*60)
    print("\nOpen your browser and navigate to:")
    print("   http://localhost:5000")
    print("\nControls:")
    print("   - Adjust nutrient, protein, and chemical levels")
    print("   - Watch cells respond in real-time")
    print("   - Rotate view with mouse drag")
    print("   - Reset simulation anytime")
    print("\n" + "="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
