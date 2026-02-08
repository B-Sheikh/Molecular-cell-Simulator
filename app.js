/**
 * 3D Cell Simulator - Main Application
 * Handles 3D visualization and user interactions
 */

class CellSimulator3D {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.cells = [];
        this.cellMeshes = {};
        this.controls = {
            nutrients: 50,
            proteins: 50,
            chemicals: 0
        };
        this.aiMode = false;
        this.chemicalMixture = [];
        this.knownChemicals = [];

        this.init();
        this.loadKnownChemicals();
        this.setupEventListeners();
        this.animate();
    }

    init() {
        const container = document.getElementById('canvas-container');
        const width = container.clientWidth;
        const height = container.clientHeight;

        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000814);

        // Create camera
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        this.camera.position.z = 8;
        this.camera.position.y = 3;
        this.camera.lookAt(0, 0, 0);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);

        // Add lights
        this.setupLighting();

        // Create initial cells
        this.createCells();

        // Add mouse controls
        this.setupMouseControls();

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 5);
        this.scene.add(directionalLight);

        // Point lights for better cell visualization
        const pointLight1 = new THREE.PointLight(0x00ffff, 0.5);
        pointLight1.position.set(-5, 5, 5);
        this.scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0xff00ff, 0.5);
        pointLight2.position.set(5, -5, 5);
        this.scene.add(pointLight2);
    }

    createCells() {
        const positions = [
            { x: -3, y: 0, z: 0 },
            { x: 0, y: 0, z: 0 },
            { x: 3, y: 0, z: 0 }
        ];

        positions.forEach((pos, index) => {
            const cellId = `cell_${index}`;

            // Create cell geometry (sphere)
            const geometry = new THREE.SphereGeometry(1, 32, 32);

            // Create material with glow effect
            const material = new THREE.MeshPhongMaterial({
                color: 0x00ff00,
                emissive: 0x00ff00,
                emissiveIntensity: 0.3,
                shininess: 100,
                transparent: true,
                opacity: 0.9
            });

            const cell = new THREE.Mesh(geometry, material);
            cell.position.set(pos.x, pos.y, pos.z);

            // Add cell membrane (outer sphere)
            const membraneGeometry = new THREE.SphereGeometry(1.1, 32, 32);
            const membraneMaterial = new THREE.MeshBasicMaterial({
                color: 0x00ff00,
                wireframe: true,
                transparent: true,
                opacity: 0.3
            });
            const membrane = new THREE.Mesh(membraneGeometry, membraneMaterial);
            cell.add(membrane);

            // Add organelles (smaller spheres inside)
            this.addOrganelles(cell);

            this.scene.add(cell);
            this.cellMeshes[cellId] = {
                mesh: cell,
                membrane: membrane,
                material: material,
                membraneMaterial: membraneMaterial
            };
        });
    }

    addOrganelles(cell) {
        // Nucleus
        const nucleusGeometry = new THREE.SphereGeometry(0.3, 16, 16);
        const nucleusMaterial = new THREE.MeshPhongMaterial({
            color: 0x4444ff,
            emissive: 0x0000ff,
            emissiveIntensity: 0.5
        });
        const nucleus = new THREE.Mesh(nucleusGeometry, nucleusMaterial);
        nucleus.position.set(0, 0, 0);
        cell.add(nucleus);

        // Mitochondria (energy centers)
        for (let i = 0; i < 3; i++) {
            const mitoGeometry = new THREE.SphereGeometry(0.15, 12, 12);
            const mitoMaterial = new THREE.MeshPhongMaterial({
                color: 0xff4444,
                emissive: 0xff0000,
                emissiveIntensity: 0.3
            });
            const mito = new THREE.Mesh(mitoGeometry, mitoMaterial);

            const angle = (i / 3) * Math.PI * 2;
            const radius = 0.5;
            mito.position.set(
                Math.cos(angle) * radius,
                Math.sin(angle * 0.5) * 0.3,
                Math.sin(angle) * radius
            );
            cell.add(mito);
        }
    }

    setupMouseControls() {
        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };
        const canvas = this.renderer.domElement;

        canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!isDragging) return;

            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;

            // Rotate camera around scene
            const rotationSpeed = 0.005;
            this.camera.position.x += deltaX * rotationSpeed;
            this.camera.position.y -= deltaY * rotationSpeed;
            this.camera.lookAt(0, 0, 0);

            previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        canvas.addEventListener('mouseup', () => {
            isDragging = false;
        });

        canvas.addEventListener('mouseleave', () => {
            isDragging = false;
        });

        // Zoom with mouse wheel
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomSpeed = 0.1;
            this.camera.position.z += e.deltaY * zoomSpeed * 0.01;
            this.camera.position.z = Math.max(3, Math.min(15, this.camera.position.z));
        });
    }

    setupEventListeners() {
        // Mode toggle
        document.querySelectorAll('input[name="mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.aiMode = e.target.value === 'ai';
                this.toggleMode();
            });
        });

        // Update slider value displays
        const sliders = ['nutrients', 'proteins', 'chemicals'];
        sliders.forEach(slider => {
            const element = document.getElementById(slider);
            const display = document.getElementById(`${slider}-value`);

            if (element && display) {
                element.addEventListener('input', (e) => {
                    display.textContent = e.target.value;
                    this.controls[slider] = parseFloat(e.target.value);
                });
            }
        });

        // Update simulation button
        document.getElementById('update-btn').addEventListener('click', () => {
            this.updateSimulation();
        });

        // Reset button
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetSimulation();
        });

        // AI mode controls
        document.getElementById('add-chemical-btn')?.addEventListener('click', () => {
            this.addChemical();
        });

        document.getElementById('chemical-name')?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.addChemical();
            }
        });

        // Preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const preset = e.target.dataset.preset;
                this.loadPreset(preset);
            });
        });
    }

    async loadKnownChemicals() {
        try {
            const response = await fetch('/api/chemicals');
            const data = await response.json();
            if (data.success) {
                this.knownChemicals = data.chemicals;
                this.populateChemicalDatalist();
            }
        } catch (error) {
            console.error('Error loading chemicals:', error);
        }
    }

    populateChemicalDatalist() {
        const datalist = document.getElementById('known-chemicals');
        if (!datalist) return;

        datalist.innerHTML = '';
        this.knownChemicals.forEach(chem => {
            const option = document.createElement('option');
            option.value = chem;
            datalist.appendChild(option);
        });
    }

    toggleMode() {
        const simpleControls = document.getElementById('simple-controls');
        const aiControls = document.getElementById('ai-controls');
        const aiAnalysis = document.getElementById('ai-analysis');

        if (this.aiMode) {
            simpleControls.style.display = 'none';
            aiControls.style.display = 'block';
            aiAnalysis.style.display = 'block';
        } else {
            simpleControls.style.display = 'block';
            aiControls.style.display = 'none';
            aiAnalysis.style.display = 'none';
        }
    }

    addChemical() {
        const nameInput = document.getElementById('chemical-name');
        const concInput = document.getElementById('chemical-concentration');

        const name = nameInput.value.trim();
        const concentration = parseFloat(concInput.value) || 50;

        if (!name) {
            alert('Please enter a chemical name');
            return;
        }

        this.chemicalMixture.push({
            name: name,
            concentration: Math.max(0, Math.min(100, concentration))
        });

        this.renderChemicalList();
        nameInput.value = '';
        concInput.value = 50;
    }

    removeChemical(index) {
        this.chemicalMixture.splice(index, 1);
        this.renderChemicalList();
    }

    renderChemicalList() {
        const container = document.getElementById('chemical-list');
        if (!container) return;

        if (this.chemicalMixture.length === 0) {
            container.innerHTML = '<p style="color: #999; font-style: italic;">No chemicals added yet</p>';
            return;
        }

        container.innerHTML = '';
        this.chemicalMixture.forEach((chem, index) => {
            const item = document.createElement('div');
            item.className = 'chemical-item';

            const isUnknown = !this.knownChemicals.includes(chem.name.toLowerCase());

            item.innerHTML = `
                <div class="chemical-item-info">
                    <div class="chemical-item-name">
                        ${chem.name}
                        ${isUnknown ? '<span class="unknown-chem-badge">Unknown</span>' : ''}
                    </div>
                    <div class="chemical-item-conc">Concentration: ${chem.concentration}%</div>
                </div>
                <button class="chemical-item-remove" onclick="app.removeChemical(${index})">Remove</button>
            `;

            container.appendChild(item);
        });
    }

    loadPreset(presetName) {
        this.chemicalMixture = [];

        switch(presetName) {
            case 'toxic':
                this.chemicalMixture = [
                    { name: 'cyanide', concentration: 60 },
                    { name: 'mercury', concentration: 45 },
                    { name: 'bleach', concentration: 30 }
                ];
                break;

            case 'healthy':
                this.chemicalMixture = [
                    { name: 'glucose', concentration: 50 },
                    { name: 'vitamins', concentration: 40 },
                    { name: 'antioxidant', concentration: 30 }
                ];
                break;

            case 'oxidative':
                this.chemicalMixture = [
                    { name: 'hydrogen_peroxide', concentration: 70 },
                    { name: 'ozone', concentration: 50 },
                    { name: 'rotenone', concentration: 40 }
                ];
                break;

            case 'unknown':
                this.chemicalMixture = [
                    { name: 'mysterious_compound_X', concentration: 55 },
                    { name: 'alien_enzyme_Z9', concentration: 60 },
                    { name: 'experimental_drug_42', concentration: 45 }
                ];
                break;
        }

        this.renderChemicalList();
    }

    async updateSimulation() {
        const button = document.getElementById('update-btn');
        button.disabled = true;
        button.innerHTML = '<span class="loading"></span> Updating...';

        try {
            let response, data;

            if (this.aiMode) {
                // Use AI endpoint
                response = await fetch('/api/simulate-ai', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        nutrients: this.controls.nutrients,
                        proteins: this.controls.proteins,
                        chemicals: this.chemicalMixture
                    })
                });
            } else {
                // Use simple endpoint
                response = await fetch('/api/simulate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(this.controls)
                });
            }

            data = await response.json();

            if (data.success) {
                this.updateCellVisuals(data.cells);
                this.displayCellStatus(data.cells);

                if (data.ai_powered) {
                    this.displayAIAnalysis(data.cells);
                }
            }
        } catch (error) {
            console.error('Simulation error:', error);
            alert('Error updating simulation. Please try again.');
        } finally {
            button.disabled = false;
            button.textContent = 'Update Simulation';
        }
    }

    displayAIAnalysis(cells) {
        const container = document.getElementById('ai-insights');
        if (!container) return;

        container.innerHTML = '';

        // Combine all analysis from cells
        const allAnalysis = [];
        const unknownChemicals = new Set();

        cells.forEach(cellData => {
            if (cellData.analysis) {
                cellData.analysis.forEach(item => {
                    if (!allAnalysis.includes(item)) {
                        allAnalysis.push(item);
                    }
                });
            }
            if (cellData.unknown_chemicals) {
                cellData.unknown_chemicals.forEach(chem => unknownChemicals.add(chem));
            }
        });

        // Display analysis
        if (allAnalysis.length > 0) {
            allAnalysis.forEach(insight => {
                const item = document.createElement('div');
                item.className = 'insight-item';
                item.textContent = insight;
                container.appendChild(item);
            });
        }

        // Display average viability
        const avgViability = cells.reduce((sum, c) => sum + (c.viability || 0), 0) / cells.length;
        const viabilityDiv = document.createElement('div');
        viabilityDiv.className = 'viability-indicator';

        let viabilityClass = 'viability-low';
        if (avgViability > 70) viabilityClass = 'viability-high';
        else if (avgViability > 40) viabilityClass = 'viability-medium';

        viabilityDiv.innerHTML = `
            <span class="${viabilityClass}">
                Average Cell Viability: ${avgViability.toFixed(1)}%
            </span>
        `;
        container.appendChild(viabilityDiv);
    }

    updateCellVisuals(cells) {
        cells.forEach(cellData => {
            const cellObj = this.cellMeshes[cellData.cell_id];
            if (!cellObj) return;

            const { mesh, material, membraneMaterial } = cellObj;

            // Animate size change
            this.animateScale(mesh, cellData.size);

            // Update color based on health
            const color = new THREE.Color(cellData.color);
            material.color = color;
            material.emissive = color;
            membraneMaterial.color = color;

            // Update emissive intensity based on health
            material.emissiveIntensity = 0.1 + (cellData.health / 100) * 0.5;

            // Update opacity based on health
            material.opacity = 0.7 + (cellData.health / 100) * 0.3;
        });
    }

    animateScale(mesh, targetScale) {
        // Smooth animation to target scale
        const currentScale = mesh.scale.x;
        const steps = 30;
        let step = 0;

        const animate = () => {
            if (step < steps) {
                const progress = step / steps;
                const newScale = currentScale + (targetScale - currentScale) * progress;
                mesh.scale.set(newScale, newScale, newScale);
                step++;
                requestAnimationFrame(animate);
            }
        };

        animate();
    }

    displayCellStatus(cells) {
        const statusContainer = document.getElementById('cell-status');
        statusContainer.innerHTML = '';

        cells.forEach((cellData, index) => {
            const cellDiv = document.createElement('div');
            cellDiv.className = 'cell-info';

            const healthColor = this.getHealthBarColor(cellData.health);

            cellDiv.innerHTML = `
                <strong>Cell ${index + 1}</strong><br>
                Health: ${cellData.health}%<br>
                Status: ${cellData.status}<br>
                Size: ${cellData.size}x
                <div class="health-bar">
                    <div class="health-fill" style="width: ${cellData.health}%; background: ${healthColor};"></div>
                </div>
            `;

            statusContainer.appendChild(cellDiv);
        });
    }

    getHealthBarColor(health) {
        if (health > 70) return '#00ff00';
        if (health > 40) return '#ffff00';
        if (health > 20) return '#ff8800';
        return '#ff0000';
    }

    async resetSimulation() {
        try {
            await fetch('/api/reset', { method: 'POST' });

            // Reset sliders
            document.getElementById('nutrients').value = 50;
            document.getElementById('proteins').value = 50;
            document.getElementById('chemicals').value = 0;

            document.getElementById('nutrients-value').textContent = '50';
            document.getElementById('proteins-value').textContent = '50';
            document.getElementById('chemicals-value').textContent = '0';

            this.controls = { nutrients: 50, proteins: 50, chemicals: 0 };

            // Reset AI mode chemicals
            this.chemicalMixture = [];
            this.renderChemicalList();

            // Clear AI analysis
            const aiInsights = document.getElementById('ai-insights');
            if (aiInsights) {
                aiInsights.innerHTML = '<p style="color: #999; font-style: italic;">Run simulation to see AI analysis</p>';
            }

            // Reset visuals
            Object.values(this.cellMeshes).forEach(cellObj => {
                cellObj.mesh.scale.set(1, 1, 1);
                const greenColor = new THREE.Color(0x00ff00);
                cellObj.material.color = greenColor;
                cellObj.material.emissive = greenColor;
                cellObj.material.emissiveIntensity = 0.3;
                cellObj.material.opacity = 0.9;
                cellObj.membraneMaterial.color = greenColor;
            });

            document.getElementById('cell-status').innerHTML = '<p>Reset complete. Adjust parameters and update simulation.</p>';
        } catch (error) {
            console.error('Reset error:', error);
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Gentle rotation animation for cells
        const time = Date.now() * 0.0005;
        Object.values(this.cellMeshes).forEach((cellObj, index) => {
            cellObj.mesh.rotation.y = time + index * 0.5;
            cellObj.mesh.rotation.x = Math.sin(time * 0.5 + index) * 0.1;
        });

        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        const container = document.getElementById('canvas-container');
        const width = container.clientWidth;
        const height = container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
}

// Initialize the application when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    console.log('🧬 Starting AI-Powered 3D Cell Simulator...');
    app = new CellSimulator3D();
});
