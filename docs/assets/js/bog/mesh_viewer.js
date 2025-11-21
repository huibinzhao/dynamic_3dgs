/**
 * PLY Mesh Viewer with Mouse Controls
 * Compatible with Chrome, Firefox, Safari, and Edge
 */

class PLYMeshViewer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }

        const sanitizedOrbit = options.initialOrbit ? { ...options.initialOrbit } : undefined;
        let derivedTarget = null;
        if (sanitizedOrbit && sanitizedOrbit.target) {
            derivedTarget = { ...sanitizedOrbit.target };
            delete sanitizedOrbit.target;
        }

        const initialTarget = options.initialTarget
            ? { ...options.initialTarget }
            : derivedTarget || { x: 0, y: 0, z: 0 };

        // Configuration options
        this.options = {
            width: options.width || this.container.clientWidth || 800,
            height: options.height || this.container.clientHeight || 600,
            backgroundColor: options.backgroundColor !== undefined ? options.backgroundColor : 0x1a1a1a,
            meshColor: options.meshColor !== undefined ? options.meshColor : 0x00aaff,
            wireframe: options.wireframe || false,
            // When true, ignore any vertex colors coming from the PLY and use meshColor instead
            forceMeshColor: options.forceMeshColor || false,
            autoRotate: options.autoRotate || false,
            autoRotateSpeed: options.autoRotateSpeed || 1.0,
            ...options,
            initialOrbit: sanitizedOrbit,
            initialTarget
        };

        this.init();
        this.setupEventListeners();
        this.animate();
    }

    init() {
        // Create scene
        this.scene = new THREE.Scene();

        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.options.width / this.options.height,
            0.1,
            1000
        );
        this.camera.position.set(0, 0, 2);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true  // Better Chrome compatibility
        });
        this.renderer.setSize(this.options.width, this.options.height);
        this.renderer.setClearColor(this.options.backgroundColor, 1);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit for performance
        this.container.appendChild(this.renderer.domElement);

        // Add lights
        this.setupLights();

        // Mouse controls
        this.setupMouseControls();

        // Store mesh reference
        this.mesh = null;

        // Animation frame ID
        this.animationId = null;
    }

    setupLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // Directional lights
        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight1.position.set(5, 5, 5);
        this.scene.add(directionalLight1);

        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        directionalLight2.position.set(-5, -5, -5);
        this.scene.add(directionalLight2);
    }

    setupMouseControls() {
        this.controls = {
            isRotating: false,
            isPanning: false,
            lastMouseX: 0,
            lastMouseY: 0,
            rotationSpeed: 0.005,
            panSpeed: 0.01,
            zoomSpeed: 0.1
        };

        this.cameraTarget = new THREE.Vector3(0, 0, 0);

        // Add momentum/inertia
        this.momentum = {
            rotation: { x: 0, y: 0 },
            pan: { x: 0, y: 0 },
            damping: 0.92,  // Higher = more weight/inertia (0-1)
            threshold: 0.001  // Stop when momentum is very small
        };
    }

    setupEventListeners() {
        const canvas = this.renderer.domElement;

        // Mouse events
        canvas.addEventListener('mousedown', (e) => this.onMouseDown(e), false);
        canvas.addEventListener('mousemove', (e) => this.onMouseMove(e), false);
        canvas.addEventListener('mouseup', (e) => this.onMouseUp(e), false);
        canvas.addEventListener('mouseleave', (e) => this.onMouseUp(e), false);
        canvas.addEventListener('wheel', (e) => this.onMouseWheel(e), { passive: false });
        canvas.addEventListener('contextmenu', (e) => e.preventDefault(), false);

        // Touch events
        canvas.addEventListener('touchstart', (e) => this.onTouchStart(e), { passive: false });
        canvas.addEventListener('touchmove', (e) => this.onTouchMove(e), { passive: false });
        canvas.addEventListener('touchend', (e) => this.onTouchEnd(e), false);

        // Window resize
        window.addEventListener('resize', () => this.onWindowResize(), false);
    }

    onMouseDown(event) {
        event.preventDefault();

        this.controls.lastMouseX = event.clientX;
        this.controls.lastMouseY = event.clientY;

        if (event.button === 0) {
            this.controls.isRotating = true;
        } else if (event.button === 2) {
            this.controls.isPanning = true;
        }
    }

    onMouseMove(event) {
        if (!this.controls.isRotating && !this.controls.isPanning) {
            return;
        }

        const deltaX = event.clientX - this.controls.lastMouseX;
        const deltaY = event.clientY - this.controls.lastMouseY;

        if (this.controls.isRotating) {
            this.rotateCamera(deltaX, deltaY);
            // Store momentum
            this.momentum.rotation.x = deltaX;
            this.momentum.rotation.y = deltaY;
        } else if (this.controls.isPanning) {
            this.panCamera(deltaX, deltaY);
            // Store momentum
            this.momentum.pan.x = deltaX;
            this.momentum.pan.y = deltaY;
        }

        this.controls.lastMouseX = event.clientX;
        this.controls.lastMouseY = event.clientY;
    }

    onMouseUp(event) {
        this.controls.isRotating = false;
        this.controls.isPanning = false;
        // Momentum will continue in animate loop
    }

    onMouseWheel(event) {
        event.preventDefault();
        const delta = event.deltaY;
        this.zoomCamera(delta);
    }

    onTouchStart(event) {
        if (event.touches.length === 1) {
            this.controls.isRotating = true;
            this.controls.lastMouseX = event.touches[0].clientX;
            this.controls.lastMouseY = event.touches[0].clientY;
        } else if (event.touches.length === 2) {
            this.controls.isPanning = true;
            this.controls.lastMouseX = (event.touches[0].clientX + event.touches[1].clientX) / 2;
            this.controls.lastMouseY = (event.touches[0].clientY + event.touches[1].clientY) / 2;
        }
    }

    onTouchMove(event) {
        event.preventDefault();

        if (event.touches.length === 1 && this.controls.isRotating) {
            const deltaX = event.touches[0].clientX - this.controls.lastMouseX;
            const deltaY = event.touches[0].clientY - this.controls.lastMouseY;
            this.rotateCamera(deltaX, deltaY);
            // Store momentum
            this.momentum.rotation.x = deltaX;
            this.momentum.rotation.y = deltaY;
            this.controls.lastMouseX = event.touches[0].clientX;
            this.controls.lastMouseY = event.touches[0].clientY;
        } else if (event.touches.length === 2 && this.controls.isPanning) {
            const centerX = (event.touches[0].clientX + event.touches[1].clientX) / 2;
            const centerY = (event.touches[0].clientY + event.touches[1].clientY) / 2;
            const deltaX = centerX - this.controls.lastMouseX;
            const deltaY = centerY - this.controls.lastMouseY;
            this.panCamera(deltaX, deltaY);
            // Store momentum
            this.momentum.pan.x = deltaX;
            this.momentum.pan.y = deltaY;
            this.controls.lastMouseX = centerX;
            this.controls.lastMouseY = centerY;
        }
    }

    onTouchEnd(event) {
        this.controls.isRotating = false;
        this.controls.isPanning = false;
    }

    rotateCamera(deltaX, deltaY) {
        const position = this.camera.position.clone().sub(this.cameraTarget);
        const radius = position.length();

        let theta = Math.atan2(position.x, position.z);
        let phi = Math.acos(Math.max(-1, Math.min(1, position.y / radius)));

        theta -= deltaX * this.controls.rotationSpeed;
        phi -= deltaY * this.controls.rotationSpeed;
        phi = Math.max(0.1, Math.min(Math.PI - 0.1, phi));

        position.x = radius * Math.sin(phi) * Math.sin(theta);
        position.y = radius * Math.cos(phi);
        position.z = radius * Math.sin(phi) * Math.cos(theta);

        this.camera.position.copy(position.add(this.cameraTarget));
        this.camera.lookAt(this.cameraTarget);
    }

    panCamera(deltaX, deltaY) {
        const distance = this.camera.position.distanceTo(this.cameraTarget);
        const panSpeed = this.controls.panSpeed * distance * 0.1;

        const right = new THREE.Vector3();
        const up = new THREE.Vector3();

        right.setFromMatrixColumn(this.camera.matrix, 0);
        up.setFromMatrixColumn(this.camera.matrix, 1);

        const offset = new THREE.Vector3();
        offset.add(right.multiplyScalar(-deltaX * panSpeed));
        offset.add(up.multiplyScalar(deltaY * panSpeed));

        this.camera.position.add(offset);
        this.cameraTarget.add(offset);
    }

    zoomCamera(delta) {
        const distance = this.camera.position.distanceTo(this.cameraTarget);
        const zoomFactor = 1 + (delta * this.controls.zoomSpeed * 0.001);

        const direction = this.camera.position.clone().sub(this.cameraTarget);
        const newDistance = distance * zoomFactor;

        if (newDistance > 0.5 && newDistance < 100) {
            direction.normalize().multiplyScalar(newDistance);
            this.camera.position.copy(this.cameraTarget.clone().add(direction));
        }
    }

    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    async loadPLY(source) {
        try {
            let plyData;

            if (typeof source === 'string') {
                const response = await fetch(source);
                if (!response.ok) {
                    throw new Error(`Failed to fetch PLY file: ${response.statusText}`);
                }
                plyData = await response.arrayBuffer();
            } else if (source instanceof File) {
                plyData = await source.arrayBuffer();
            } else if (source instanceof ArrayBuffer) {
                plyData = source;
            } else {
                throw new Error('Invalid source type for PLY file');
            }

            const geometry = this.parsePLY(plyData);
            this.displayMesh(geometry);
        } catch (error) {
            console.error('Error loading PLY file:', error);
            throw error;
        }
    }

    parsePLY(arrayBuffer) {
        const data = new Uint8Array(arrayBuffer);
        let position = 0;

        // Read header
        const decoder = new TextDecoder('ascii');
        let headerText = '';

        while (position < data.length) {
            const char = String.fromCharCode(data[position]);
            headerText += char;
            position++;

            if (headerText.endsWith('end_header\n') || headerText.endsWith('end_header\r\n')) {
                break;
            }
        }

        // Parse header
        const lines = headerText.split('\n').map(line => line.trim());
        let vertexCount = 0;
        let faceCount = 0;
        let format = 'ascii';
        const vertexProperties = [];
        let inVertex = false;
        let inFace = false;

        for (const line of lines) {
            const tokens = line.split(/\s+/);

            if (tokens[0] === 'format') {
                format = tokens[1];
            } else if (tokens[0] === 'element') {
                if (tokens[1] === 'vertex') {
                    vertexCount = parseInt(tokens[2]);
                    inVertex = true;
                    inFace = false;
                } else if (tokens[1] === 'face') {
                    faceCount = parseInt(tokens[2]);
                    inFace = true;
                    inVertex = false;
                }
            } else if (tokens[0] === 'property') {
                if (inVertex) {
                    vertexProperties.push({ type: tokens[1], name: tokens[2] });
                }
            }
        }

        const geometry = new THREE.BufferGeometry();

        if (format.startsWith('binary')) {
            this.parseBinaryPLY(data, position, vertexCount, faceCount, vertexProperties, geometry, format);
        } else {
            this.parseAsciiPLY(headerText, arrayBuffer, vertexCount, faceCount, geometry);
        }

        geometry.computeVertexNormals();
        return geometry;
    }

    parseBinaryPLY(data, position, vertexCount, faceCount, vertexProperties, geometry, format) {
        const littleEndian = format.includes('little');
        const view = new DataView(data.buffer, position);
        let offset = 0;

        const vertices = [];
        const colors = [];
        const hasColor = vertexProperties.some(p => p.name === 'red' || p.name === 'r');

        for (let i = 0; i < vertexCount; i++) {
            let x = 0, y = 0, z = 0;
            let r = 255, g = 255, b = 255;

            for (const prop of vertexProperties) {
                let value;

                if (prop.type === 'float') {
                    value = view.getFloat32(offset, littleEndian);
                    offset += 4;
                } else if (prop.type === 'double') {
                    value = view.getFloat64(offset, littleEndian);
                    offset += 8;
                } else if (prop.type === 'uchar') {
                    value = view.getUint8(offset);
                    offset += 1;
                } else if (prop.type === 'int') {
                    value = view.getInt32(offset, littleEndian);
                    offset += 4;
                }

                if (prop.name === 'x') x = value;
                else if (prop.name === 'y') y = value;
                else if (prop.name === 'z') z = value;
                else if (prop.name === 'red' || prop.name === 'r') r = value;
                else if (prop.name === 'green' || prop.name === 'g') g = value;
                else if (prop.name === 'blue' || prop.name === 'b') b = value;
            }

            vertices.push(x, y, z);
            if (hasColor) {
                colors.push(r / 255, g / 255, b / 255);
            }
        }

        const indices = [];

        for (let i = 0; i < faceCount; i++) {
            const numVertices = view.getUint8(offset);
            offset += 1;

            const faceIndices = [];
            for (let j = 0; j < numVertices; j++) {
                faceIndices.push(view.getInt32(offset, littleEndian));
                offset += 4;
            }

            if (numVertices === 3) {
                indices.push(faceIndices[0], faceIndices[1], faceIndices[2]);
            } else if (numVertices === 4) {
                indices.push(faceIndices[0], faceIndices[1], faceIndices[2]);
                indices.push(faceIndices[0], faceIndices[2], faceIndices[3]);
            }
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setIndex(indices);

        if (hasColor && colors.length > 0) {
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        }
    }

    parseAsciiPLY(headerText, arrayBuffer, vertexCount, faceCount, geometry) {
        const decoder = new TextDecoder('ascii');
        const text = decoder.decode(arrayBuffer);
        const lines = text.split('\n');

        let dataStartLine = 0;
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].trim() === 'end_header') {
                dataStartLine = i + 1;
                break;
            }
        }

        const vertices = [];
        const colors = [];

        for (let i = 0; i < vertexCount; i++) {
            const line = lines[dataStartLine + i].trim();
            if (!line) continue;

            const values = line.split(/\s+/).map(v => parseFloat(v));
            vertices.push(values[0], values[1], values[2]);

            if (values.length >= 6) {
                colors.push(values[3] / 255, values[4] / 255, values[5] / 255);
            }
        }

        const indices = [];
        for (let i = 0; i < faceCount; i++) {
            const line = lines[dataStartLine + vertexCount + i].trim();
            if (!line) continue;

            const values = line.split(/\s+/).map(v => parseInt(v));
            const numVertices = values[0];

            if (numVertices === 3) {
                indices.push(values[1], values[2], values[3]);
            } else if (numVertices === 4) {
                indices.push(values[1], values[2], values[3]);
                indices.push(values[1], values[3], values[4]);
            }
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setIndex(indices);

        if (colors.length > 0) {
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        }
    }

    displayMesh(geometry) {
        if (this.mesh) {
            this.scene.remove(this.mesh);
            if (this.mesh.geometry) this.mesh.geometry.dispose();
            if (this.mesh.material) this.mesh.material.dispose();
        }

        const hasColors = geometry.attributes.color !== undefined && !this.options.forceMeshColor;
        const material = new THREE.MeshStandardMaterial({
            color: this.options.meshColor,
            vertexColors: hasColors,
            wireframe: this.options.wireframe,
            metalness: 0.3,
            roughness: 0.7,
            side: THREE.DoubleSide,
            flatShading: false
        });

        this.mesh = new THREE.Mesh(geometry, material);
        this.scene.add(this.mesh);

        this.centerMesh();
    }

    centerMesh() {
        if (!this.mesh) return;

        this.mesh.geometry.computeBoundingBox();
        const boundingBox = this.mesh.geometry.boundingBox;

        const center = new THREE.Vector3();
        boundingBox.getCenter(center);
        this.mesh.geometry.translate(-center.x, -center.y, -center.z);

        const size = new THREE.Vector3();
        boundingBox.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 4 / maxDim;
        this.mesh.scale.set(scale, scale, scale);

        const target = this.options.initialTarget || { x: 0, y: 0, z: 0 };
        this.cameraTarget.set(target.x || 0, target.y || 0, target.z || 0);
        // Recompute a comfortable camera distance based on the scaled mesh size
        const scaledSize = size.clone().multiplyScalar(scale);
        const scaledMaxDim = Math.max(scaledSize.x, scaledSize.y, scaledSize.z);
        const boundingRadius = scaledMaxDim * 0.5;
        const fovInRad = THREE.MathUtils.degToRad(this.camera.fov);
        const distance = (boundingRadius / Math.sin(fovInRad / 2)) * 1.2; // add slight padding

        // Prevent the camera from getting too close or too far
        const clampedDistance = THREE.MathUtils.clamp(distance, 1.0, 20.0);
        this.applyOrbit(clampedDistance, this.options.initialOrbit);
    }

    applyOrbit(distance, orbitConfig = {}) {
        const orbit = {
            theta: 0,
            phi: Math.PI / 2,
            radiusScale: 1,
            radius: undefined,
            ...orbitConfig
        };

        const effectiveDistance = orbit.radius !== undefined
            ? orbit.radius
            : distance * orbit.radiusScale;

        const sinPhi = Math.sin(orbit.phi);
        const cosPhi = Math.cos(orbit.phi);
        const sinTheta = Math.sin(orbit.theta);
        const cosTheta = Math.cos(orbit.theta);

        const x = effectiveDistance * sinPhi * sinTheta;
        const y = effectiveDistance * cosPhi;
        const z = effectiveDistance * sinPhi * cosTheta;

        this.camera.position.set(x, y, z);
        this.camera.lookAt(this.cameraTarget);
    }

    getCurrentOrbit() {
        const relative = this.camera.position.clone().sub(this.cameraTarget);
        const radius = relative.length();

        if (radius === 0) {
            return { theta: 0, phi: 0, radius: 0 };
        }

        const theta = Math.atan2(relative.x, relative.z);
        const yRatio = THREE.MathUtils.clamp(relative.y / radius, -1, 1);
        const phi = Math.acos(yRatio);

        return { theta, phi, radius };
    }

    getCurrentTarget() {
        return { x: this.cameraTarget.x, y: this.cameraTarget.y, z: this.cameraTarget.z };
    }

    getCurrentViewState() {
        return {
            ...this.getCurrentOrbit(),
            target: this.getCurrentTarget()
        };
    }

    setWireframe(enabled) {
        if (this.mesh && this.mesh.material) {
            this.mesh.material.wireframe = enabled;
        }
    }

    setMeshColor(color) {
        if (this.mesh && this.mesh.material) {
            this.mesh.material.color.set(color);
        }
    }

    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());

        // Apply momentum when not actively dragging
        if (!this.controls.isRotating && !this.controls.isPanning) {
            // Apply rotation momentum
            if (Math.abs(this.momentum.rotation.x) > this.momentum.threshold ||
                Math.abs(this.momentum.rotation.y) > this.momentum.threshold) {
                this.rotateCamera(this.momentum.rotation.x, this.momentum.rotation.y);
                this.momentum.rotation.x *= this.momentum.damping;
                this.momentum.rotation.y *= this.momentum.damping;
            }

            // Apply pan momentum
            if (Math.abs(this.momentum.pan.x) > this.momentum.threshold ||
                Math.abs(this.momentum.pan.y) > this.momentum.threshold) {
                this.panCamera(this.momentum.pan.x, this.momentum.pan.y);
                this.momentum.pan.x *= this.momentum.damping;
                this.momentum.pan.y *= this.momentum.damping;
            }
        } else {
            // Reset momentum when actively dragging
            if (this.controls.isRotating) {
                // Keep rotation momentum building while dragging
            } else if (this.controls.isPanning) {
                // Keep pan momentum building while dragging
            }
        }

        // Auto-rotate
        if (this.options.autoRotate && this.mesh) {
            this.mesh.rotation.y += 0.01 * this.options.autoRotateSpeed;
        }

        this.renderer.render(this.scene, this.camera);
    }

    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        if (this.mesh) {
            this.scene.remove(this.mesh);
            if (this.mesh.geometry) this.mesh.geometry.dispose();
            if (this.mesh.material) this.mesh.material.dispose();
        }

        this.renderer.dispose();
        if (this.container && this.renderer.domElement.parentNode === this.container) {
            this.container.removeChild(this.renderer.domElement);
        }
    }
}

// Make it available globally
if (typeof window !== 'undefined') {
    window.PLYMeshViewer = PLYMeshViewer;
}
