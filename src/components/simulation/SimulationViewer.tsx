import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { SimulationFrame } from '../../services/api';

interface Props {
  frame: SimulationFrame | null;
  width?: number;
  height?: number;
  corridorLength?: number;
  corridorWidth?: number;
  corridorHeight?: number;
  agentDiameter?: number;
  agentMaxSpeed?: number;
}

// Swarm drone class - visualization only, positions controlled by trajectory data
class EnemyDrone {
  mesh: THREE.Group;
  diameter: number;
  speed: number;
  id: number;
  velocityArrow: THREE.ArrowHelper | null = null;

  constructor(scene: THREE.Scene, position: THREE.Vector3, diameter: number, speed: number, id: number) {
    this.diameter = diameter;
    this.speed = speed;
    this.id = id;

    // Create a detailed quadcopter for swarm drones
    const droneGroup = new THREE.Group();

    // Central body - more realistic shape
    const bodyGeometry = new THREE.BoxGeometry(diameter * 0.6, diameter * 0.3, diameter * 0.6);
    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: 0xff0000,
      emissive: 0xff0000,
      emissiveIntensity: 0.4,
      metalness: 0.7,
      roughness: 0.3,
    });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.castShadow = true;
    droneGroup.add(body);

    // Add 4 arms in X configuration
    const armLength = diameter * 0.7;
    const armGeometry = new THREE.BoxGeometry(armLength, diameter * 0.08, diameter * 0.08);
    const armMaterial = new THREE.MeshStandardMaterial({
      color: 0x330000,
      metalness: 0.8,
      roughness: 0.2,
    });

    const armAngles = [45, 135, 225, 315];
    armAngles.forEach((angle, idx) => {
      const arm = new THREE.Mesh(armGeometry, armMaterial);
      const radians = (angle * Math.PI) / 180;
      const distance = armLength / 2;
      arm.position.set(
        Math.cos(radians) * distance,
        0,
        Math.sin(radians) * distance
      );
      arm.rotation.y = radians;
      droneGroup.add(arm);

      // Motor at end of arm
      const motorGeometry = new THREE.CylinderGeometry(diameter * 0.15, diameter * 0.15, diameter * 0.2, 12);
      const motorMaterial = new THREE.MeshStandardMaterial({
        color: 0x660000,
        metalness: 0.9,
        roughness: 0.1,
      });
      const motor = new THREE.Mesh(motorGeometry, motorMaterial);
      motor.position.set(
        Math.cos(radians) * armLength,
        diameter * 0.1,
        Math.sin(radians) * armLength
      );
      droneGroup.add(motor);

      // Propeller (spinning disc)
      const propGeometry = new THREE.CylinderGeometry(diameter * 0.35, diameter * 0.35, diameter * 0.02, 16);
      const propMaterial = new THREE.MeshStandardMaterial({
        color: 0xff4444,
        emissive: 0xff0000,
        emissiveIntensity: 0.2,
        transparent: true,
        opacity: 0.7,
        side: THREE.DoubleSide,
      });
      const prop = new THREE.Mesh(propGeometry, propMaterial);
      prop.position.set(
        Math.cos(radians) * armLength,
        diameter * 0.25,
        Math.sin(radians) * armLength
      );
      droneGroup.add(prop);
    });

    droneGroup.position.copy(position);
    scene.add(droneGroup);
    this.mesh = droneGroup;
  }

  setPosition(position: THREE.Vector3) {
    this.mesh.position.copy(position);
  }

  updateVelocity(velocity: { x: number; y: number; z: number } | null) {
    // Remove old arrow if it exists
    if (this.velocityArrow) {
      this.mesh.remove(this.velocityArrow);
      this.velocityArrow.dispose();
      this.velocityArrow = null;
    }

    if (!velocity) return;

    const velocityVec = new THREE.Vector3(velocity.x, velocity.y, velocity.z);
    const speed = velocityVec.length();

    // Only show arrow if drone is moving
    if (speed > 0.1) {
      const direction = velocityVec.clone().normalize();
      const arrowLength = Math.min(speed / this.speed, 1) * this.diameter * 2;
      const arrowColor = 0xff8800; // Orange for swarm drone velocity

      const arrow = new THREE.ArrowHelper(
        direction,
        new THREE.Vector3(0, 0, 0),
        arrowLength,
        arrowColor,
        arrowLength * 0.2,
        arrowLength * 0.15
      );

      this.mesh.add(arrow);
      this.velocityArrow = arrow;
    }
  }

  dispose(scene: THREE.Scene) {
    scene.remove(this.mesh);
    // Dispose all geometries and materials in the group
    this.mesh.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(mat => mat.dispose());
          } else {
            child.material.dispose();
          }
        }
      }
    });
  }
}

function createDrone(propellersRef: React.MutableRefObject<THREE.Group[]>, motorsRef: React.MutableRefObject<THREE.Mesh[]>, diameter: number = 0.6) {
  const droneGroup = new THREE.Group();

  // Calculate scale factor based on diameter (default 0.6)
  const scale = diameter / 0.6;

  // Central body (bottom plate) - scaled based on agent diameter
  const bodyGeometry = new THREE.BoxGeometry(diameter * 0.67, diameter * 0.25, diameter * 0.67);
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: 0x1a1a1a,
    metalness: 0.7,
    roughness: 0.3,
  });
  const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
  body.castShadow = true;
  droneGroup.add(body);

  // Top plate - scaled to drone size
  const topPlateGeometry = new THREE.BoxGeometry(diameter * 0.58, diameter * 0.05, diameter * 0.58);
  const topPlateMaterial = new THREE.MeshStandardMaterial({
    color: 0x00ffff,
    emissive: 0x00ffff,
    emissiveIntensity: 0.3,
    metalness: 0.9,
    roughness: 0.1,
  });
  const topPlate = new THREE.Mesh(topPlateGeometry, topPlateMaterial);
  topPlate.position.y = 0.09 * scale;
  droneGroup.add(topPlate);

  // Bottom plate - scaled to drone size
  const bottomPlate = new THREE.Mesh(topPlateGeometry, bodyMaterial);
  bottomPlate.position.y = -0.09 * scale;
  droneGroup.add(bottomPlate);

  // LEDs - scaled to drone size
  const ledGeometry = new THREE.SphereGeometry(0.03 * scale, 8, 8);
  const frontLEDMaterial = new THREE.MeshStandardMaterial({
    color: 0x00ffff,
    emissive: 0x00ffff,
    emissiveIntensity: 2,
  });
  const frontLED = new THREE.Mesh(ledGeometry, frontLEDMaterial);
  frontLED.position.set(0, 0.12 * scale, 0.15 * scale);
  droneGroup.add(frontLED);

  const rearLEDMaterial = new THREE.MeshStandardMaterial({
    color: 0xff0000,
    emissive: 0xff0000,
    emissiveIntensity: 2,
  });
  const rearLED = new THREE.Mesh(ledGeometry, rearLEDMaterial);
  rearLED.position.set(0, 0.12 * scale, -0.15 * scale);
  droneGroup.add(rearLED);

  // Arms and propellers
  const armMaterial = new THREE.MeshStandardMaterial({
    color: 0x2a2a2a,
    metalness: 0.8,
    roughness: 0.2,
  });

  const motorMaterial = new THREE.MeshStandardMaterial({
    color: 0x0088ff,
    metalness: 0.9,
    roughness: 0.1,
  });

  const propMaterialCW = new THREE.MeshStandardMaterial({
    color: 0x00ffff,
    emissive: 0x00ffff,
    emissiveIntensity: 0.2,
    metalness: 0.5,
    roughness: 0.3,
    transparent: true,
    opacity: 0.8,
    side: THREE.DoubleSide,
  });

  const propMaterialCCW = new THREE.MeshStandardMaterial({
    color: 0x00ddff,
    emissive: 0x00ddff,
    emissiveIntensity: 0.2,
    metalness: 0.5,
    roughness: 0.3,
    transparent: true,
    opacity: 0.8,
    side: THREE.DoubleSide,
  });

  // Arm positions scaled to drone diameter
  const armOffset = diameter * 0.83; // 0.5/0.6 = 0.83
  const armPositions = [
    { pos: [armOffset, 0, armOffset], rot: Math.PI / 4, cw: false },
    { pos: [armOffset, 0, -armOffset], rot: -Math.PI / 4, cw: true },
    { pos: [-armOffset, 0, armOffset], rot: (3 * Math.PI) / 4, cw: true },
    { pos: [-armOffset, 0, -armOffset], rot: (-3 * Math.PI) / 4, cw: false },
  ];

  armPositions.forEach((armData) => {
    const armGroup = new THREE.Group();

    // Arm - scaled to drone size
    const armGeometry = new THREE.CylinderGeometry(0.025 * scale, 0.025 * scale, diameter, 8);
    const arm = new THREE.Mesh(armGeometry, armMaterial);
    arm.rotation.z = Math.PI / 2;
    arm.castShadow = true;
    armGroup.add(arm);

    // Motor - scaled to drone size
    const motorGeometry = new THREE.CylinderGeometry(0.08 * scale, 0.08 * scale, 0.12 * scale, 16);
    const motor = new THREE.Mesh(motorGeometry, motorMaterial);
    motor.position.set(diameter / 2, 0, 0);
    motor.castShadow = true;
    armGroup.add(motor);

    // Motor bell - scaled to drone size
    const motorBellGeometry = new THREE.CylinderGeometry(0.09 * scale, 0.06 * scale, 0.08 * scale, 16);
    const motorBellMaterial = new THREE.MeshStandardMaterial({
      color: 0x00aaff,
      metalness: 1.0,
      roughness: 0.05,
    });
    const motorBell = new THREE.Mesh(motorBellGeometry, motorBellMaterial);
    motorBell.position.set(diameter / 2, 0.1 * scale, 0);
    armGroup.add(motorBell);
    motorsRef.current.push(motorBell);

    // Motor shaft - scaled to drone size
    const shaftGeometry = new THREE.CylinderGeometry(0.015 * scale, 0.015 * scale, 0.06 * scale, 8);
    const shaft = new THREE.Mesh(shaftGeometry, motorMaterial);
    shaft.position.set(diameter / 2, 0.17 * scale, 0);
    armGroup.add(shaft);

    // Propeller hub - scaled to drone size
    const hubGeometry = new THREE.CylinderGeometry(0.04 * scale, 0.04 * scale, 0.02 * scale, 8);
    const hub = new THREE.Mesh(hubGeometry, motorBellMaterial);
    hub.position.set(diameter / 2, 0.21 * scale, 0);
    armGroup.add(hub);

    // Propeller group - scaled to drone size
    const propGroup = new THREE.Group();
    propGroup.position.set(diameter / 2, 0.22 * scale, 0);

    // Create 2 blades - scaled to drone size
    for (let i = 0; i < 2; i++) {
      const bladeShape = new THREE.Shape();
      bladeShape.moveTo(0, 0);
      bladeShape.quadraticCurveTo(0.15 * scale, 0.05 * scale, diameter / 2, 0.01 * scale);
      bladeShape.lineTo(diameter / 2, -0.01 * scale);
      bladeShape.quadraticCurveTo(0.15 * scale, -0.05 * scale, 0, 0);

      const bladeGeometry = new THREE.ExtrudeGeometry(bladeShape, {
        depth: 0.003 * scale,
        bevelEnabled: true,
        bevelThickness: 0.001 * scale,
        bevelSize: 0.001 * scale,
        bevelSegments: 2,
      });

      const blade = new THREE.Mesh(bladeGeometry, armData.cw ? propMaterialCW : propMaterialCCW);
      blade.rotation.y = i * Math.PI;
      blade.rotation.x = Math.PI / 2;
      blade.position.z = -0.0015 * scale;
      propGroup.add(blade);
    }

    armGroup.add(propGroup);
    propellersRef.current.push(propGroup);

    armGroup.rotation.y = armData.rot;
    droneGroup.add(armGroup);

    // Landing gear - scaled to drone size
    const legGeometry = new THREE.CylinderGeometry(0.015 * scale, 0.015 * scale, 0.25 * scale, 6);
    const legMaterial = new THREE.MeshStandardMaterial({
      color: 0x333333,
      metalness: 0.5,
      roughness: 0.5,
    });
    const leg = new THREE.Mesh(legGeometry, legMaterial);
    leg.position.set(armData.pos[0] * diameter * 0.67, -0.2 * scale, armData.pos[2] * diameter * 0.67);
    droneGroup.add(leg);

    // Footpad - scaled to drone size
    const footGeometry = new THREE.SphereGeometry(0.03 * scale, 8, 8);
    const foot = new THREE.Mesh(footGeometry, legMaterial);
    foot.position.set(armData.pos[0] * diameter * 0.67, -0.33 * scale, armData.pos[2] * diameter * 0.67);
    droneGroup.add(foot);
  });

  // Camera gimbal - scaled to drone size
  const gimbalMountGeometry = new THREE.CylinderGeometry(0.06 * scale, 0.06 * scale, 0.08 * scale, 8);
  const gimbalMount = new THREE.Mesh(gimbalMountGeometry, bodyMaterial);
  gimbalMount.position.set(0, -0.15 * scale, 0.1 * scale);
  droneGroup.add(gimbalMount);

  const cameraBodyGeometry = new THREE.BoxGeometry(0.08 * scale, 0.06 * scale, 0.06 * scale);
  const cameraMaterial = new THREE.MeshStandardMaterial({
    color: 0x1a1a1a,
    metalness: 0.6,
    roughness: 0.4,
  });
  const cameraBody = new THREE.Mesh(cameraBodyGeometry, cameraMaterial);
  cameraBody.position.set(0, -0.19 * scale, 0.13 * scale);
  droneGroup.add(cameraBody);

  const lensGeometry = new THREE.CylinderGeometry(0.02 * scale, 0.025 * scale, 0.03 * scale, 16);
  const lensMaterial = new THREE.MeshStandardMaterial({
    color: 0x000000,
    metalness: 1,
    roughness: 0.1,
  });
  const lens = new THREE.Mesh(lensGeometry, lensMaterial);
  lens.rotation.x = Math.PI / 2;
  lens.position.set(0, -0.19 * scale, 0.16 * scale);
  droneGroup.add(lens);

  // Antenna - scaled to drone size
  const antennaGeometry = new THREE.CylinderGeometry(0.008 * scale, 0.008 * scale, 0.3 * scale, 6);
  const antennaMaterial = new THREE.MeshStandardMaterial({
    color: 0x00ffff,
    emissive: 0x00ffff,
    emissiveIntensity: 0.5,
  });
  const antenna = new THREE.Mesh(antennaGeometry, antennaMaterial);
  antenna.position.set(0, 0.23 * scale, -0.1 * scale);
  antenna.rotation.x = Math.PI / 12;
  droneGroup.add(antenna);

  return droneGroup;
}

export default function SimulationViewer({ 
  frame, 
  width = 800, 
  height = 500,
  corridorLength,
  corridorWidth,
  corridorHeight,
  agentDiameter = 0.6,
  agentMaxSpeed = 10.0
}: Props) {
  // Debug: Log corridor dimensions and agent config on every render
  console.log('SimulationViewer render:', { corridorLength, corridorWidth, corridorHeight, agentDiameter, agentMaxSpeed });
  
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const droneRef = useRef<THREE.Group | null>(null);
  const targetRef = useRef<THREE.Mesh | null>(null);
  const targetLightRef = useRef<THREE.PointLight | null>(null);
  const propellersRef = useRef<THREE.Group[]>([]);
  const motorsRef = useRef<THREE.Mesh[]>([]);
  const enemiesRef = useRef<EnemyDrone[]>([]);
  const lidarRaysRef = useRef<THREE.Line[]>([]);
  const velocityArrowRef = useRef<THREE.ArrowHelper | null>(null);
  const [fps, setFps] = useState(0);
  const [isFirstPersonView, setIsFirstPersonView] = useState(false);
  const isFirstPersonViewRef = useRef(false); // Use ref for animation loop
  const frameCountRef = useRef(0);
  const fpsIntervalRef = useRef<number>();
  const animationFrameRef = useRef<number>();
  const lastTimeRef = useRef<number>(performance.now());
  const cameraOffsetRef = useRef(new THREE.Vector3(-5, 3, -5));
  const firstPersonOffsetRef = useRef(new THREE.Vector3(0, 0.3, 0.1)); // Camera offset for first-person view
  const targetPulseRef = useRef(0);

  // Sync state to ref for use in animation loop
  useEffect(() => {
    isFirstPersonViewRef.current = isFirstPersonView;
  }, [isFirstPersonView]);

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return;
    
    // Validate corridor dimensions
    if (!corridorLength || !corridorWidth || !corridorHeight) {
      console.error('SimulationViewer: Missing corridor dimensions', {
        corridorLength,
        corridorWidth,
        corridorHeight
      });
      return;
    }
    
    console.log(`[SimulationViewer] Creating scene with corridor: L=${corridorLength}, W=${corridorWidth}, H=${corridorHeight}`);
    console.log(`[SimulationViewer] Corridor bounds: X=[${-corridorWidth/2}, ${corridorWidth/2}], Y=[0, ${corridorHeight}], Z=[0, ${corridorLength}]`);

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000510);
    scene.fog = new THREE.FogExp2(0x000510, 0.02);
    sceneRef.current = scene;

    // Camera - positioned to view the start zone and corridor
    const camera = new THREE.PerspectiveCamera(
      75,
      width / height,
      0.1,
      corridorLength + 20
    );
    // Position camera to the side and above, looking at the start zone
    camera.position.set(-corridorWidth * 0.8, corridorHeight * 1.5, corridorLength * 0.15);
    camera.lookAt(0, corridorHeight / 2, corridorLength * 0.2);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // OrbitControls for camera rotation
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; // Smooth camera movements
    controls.dampingFactor = 0.05;
    controls.enableZoom = true;
    controls.enablePan = true;
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN
    };
    // Set rotation around the center of the corridor
    controls.target.set(0, corridorHeight / 2, corridorLength * 0.4);
    controls.update();
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x0077ff, 0.3);
    scene.add(ambientLight);

    const pointLight1 = new THREE.PointLight(0x00ffff, 1, 50);
    pointLight1.position.set(0, 20, 0);
    pointLight1.castShadow = true;
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0x00ffff, 1, 50);
    pointLight2.position.set(0, 20, 50);
    scene.add(pointLight2);

    const pointLight3 = new THREE.PointLight(0x00ffff, 1, 50);
    pointLight3.position.set(0, 20, 100);
    scene.add(pointLight3);

    // Floor - PlaneGeometry(width, height) where width=X, height=Z after rotation
    const floorGeometry = new THREE.PlaneGeometry(corridorWidth, corridorLength);
    const floorMaterial = new THREE.MeshStandardMaterial({
      color: 0x001a33,
      metalness: 0.7,
      roughness: 0.3,
    });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(0, 0, corridorLength / 2);
    floor.receiveShadow = true;
    scene.add(floor);

    // Create custom grid that matches corridor dimensions (width x length)
    const gridHelper = new THREE.GridHelper(corridorLength, 50, 0x00ffff, 0x003366);
    // Scale the grid to match corridor width (GridHelper is square by default)
    gridHelper.scale.set(corridorWidth / corridorLength, 1, 1);
    gridHelper.position.set(0, 0.01, corridorLength / 2);
    scene.add(gridHelper);

    // Add edge lines to mark corridor boundaries (instead of solid walls)
    const edgeMaterial = new THREE.LineBasicMaterial({ 
      color: 0x00ffff, 
      linewidth: 2,
      transparent: true,
      opacity: 0.8 
    });

    // Left edge
    const leftEdgeGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-corridorWidth / 2, 0, 0),
      new THREE.Vector3(-corridorWidth / 2, 0, corridorLength),
      new THREE.Vector3(-corridorWidth / 2, corridorHeight, corridorLength),
      new THREE.Vector3(-corridorWidth / 2, corridorHeight, 0),
      new THREE.Vector3(-corridorWidth / 2, 0, 0)
    ]);
    const leftEdge = new THREE.Line(leftEdgeGeometry, edgeMaterial);
    scene.add(leftEdge);

    // Right edge
    const rightEdgeGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(corridorWidth / 2, 0, 0),
      new THREE.Vector3(corridorWidth / 2, 0, corridorLength),
      new THREE.Vector3(corridorWidth / 2, corridorHeight, corridorLength),
      new THREE.Vector3(corridorWidth / 2, corridorHeight, 0),
      new THREE.Vector3(corridorWidth / 2, 0, 0)
    ]);
    const rightEdge = new THREE.Line(rightEdgeGeometry, edgeMaterial);
    scene.add(rightEdge);

    // Top edges
    const topFrontGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-corridorWidth / 2, corridorHeight, 0),
      new THREE.Vector3(corridorWidth / 2, corridorHeight, 0)
    ]);
    const topFront = new THREE.Line(topFrontGeometry, edgeMaterial);
    scene.add(topFront);

    const topBackGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-corridorWidth / 2, corridorHeight, corridorLength),
      new THREE.Vector3(corridorWidth / 2, corridorHeight, corridorLength)
    ]);
    const topBack = new THREE.Line(topBackGeometry, edgeMaterial);
    scene.add(topBack);

    // Create drone with actual agent diameter - position will be set by first frame
    const droneGroup = createDrone(propellersRef, motorsRef, agentDiameter);
    // Start at safe center position within corridor bounds
    droneGroup.position.set(0, corridorHeight / 2, corridorLength * 0.1);
    droneGroup.visible = false; // Hide until first frame arrives
    scene.add(droneGroup);
    droneRef.current = droneGroup;

    // Create target
    const targetGeometry = new THREE.SphereGeometry(1, 32, 32);
    const targetMaterial = new THREE.MeshStandardMaterial({
      color: 0x00ff00,
      emissive: 0x00ff00,
      emissiveIntensity: 1,
      metalness: 0.8,
      roughness: 0.2,
      transparent: true,
      opacity: 0.8,
    });
    const target = new THREE.Mesh(targetGeometry, targetMaterial);
    target.position.set(0, corridorHeight / 2, corridorLength * 0.9);
    target.visible = false; // Hide until first frame arrives with goal position
    scene.add(target);
    targetRef.current = target;

    const targetLight = new THREE.PointLight(0x00ff00, 2, 10);
    targetLight.position.copy(target.position);
    targetLight.visible = false; // Hide until target is visible
    scene.add(targetLight);
    targetLightRef.current = targetLight;

    // Enemy drones will be created based on first frame's obstacle data
    // (initialized empty, populated when first frame is received)

    // FPS counter
    fpsIntervalRef.current = window.setInterval(() => {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
    }, 1000);

    // Animation loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);

      const currentTime = performance.now();
      const deltaTime = Math.min((currentTime - lastTimeRef.current) / 1000, 0.1);
      lastTimeRef.current = currentTime;

      // Rotate propellers
      propellersRef.current.forEach((propeller, index) => {
        const direction = index === 1 || index === 2 ? 1 : -1;
        propeller.rotation.y += 0.8 * direction;
      });

      // Rotate motor bells
      motorsRef.current.forEach((motor, index) => {
        const direction = index === 1 || index === 2 ? 1 : -1;
        motor.rotation.y += 0.6 * direction;
      });

      // Note: Enemy drone positions are updated from frame data, not here

      // Animate target - no pulsing
      if (targetRef.current && targetLightRef.current) {
        // No animation - keep constant
        targetLightRef.current.intensity = 2;
      }

      // Update camera based on view mode
      if (droneRef.current && cameraRef.current && controlsRef.current) {
        const dronePos = droneRef.current.position;

        if (isFirstPersonViewRef.current) {
          // First-person view: Camera attached directly to drone
          // Disable OrbitControls in first-person mode
          controlsRef.current.enabled = false;
          
          // Calculate camera position relative to drone's orientation
          const offset = firstPersonOffsetRef.current.clone();
          offset.applyQuaternion(droneRef.current.quaternion);
          
          // Attach camera directly to drone without smoothing
          cameraRef.current.position.set(
            dronePos.x + offset.x,
            dronePos.y + offset.y,
            dronePos.z + offset.z
          );

          // Look in the direction the drone is facing (forward along Z-axis)
          const lookDistance = 5;
          const lookDirection = new THREE.Vector3(0, 0, lookDistance);
          lookDirection.applyQuaternion(droneRef.current.quaternion);
          const lookAtPoint = new THREE.Vector3(
            dronePos.x + lookDirection.x,
            dronePos.y + lookDirection.y,
            dronePos.z + lookDirection.z
          );
          
          cameraRef.current.lookAt(lookAtPoint);
        } else {
          // Third-person view: OrbitControls enabled
          controlsRef.current.enabled = true;
          controlsRef.current.update();
        }
      } else if (controlsRef.current) {
        // No drone, just update controls
        controlsRef.current.update();
      }

      renderer.render(scene, camera);
      frameCountRef.current++;
    };
    animate();

    // Cleanup
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (fpsIntervalRef.current) {
        clearInterval(fpsIntervalRef.current);
      }
      if (controlsRef.current) {
        controlsRef.current.dispose();
      }
      enemiesRef.current.forEach((enemy) => {
        enemy.dispose(scene);
      });
      enemiesRef.current = [];
      
      // Clean up LIDAR rays
      lidarRaysRef.current.forEach((ray) => {
        scene.remove(ray);
        ray.geometry.dispose();
        (ray.material as THREE.Material).dispose();
      });
      lidarRaysRef.current = [];
      
      if (containerRef.current && renderer.domElement) {
        containerRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [width, height, corridorLength, corridorWidth, corridorHeight]);

  // Keyboard controls for camera and view switching
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!cameraRef.current || !controlsRef.current) return;

      const moveSpeed = 2;
      const camera = cameraRef.current;
      const controls = controlsRef.current;

      switch (event.key.toLowerCase()) {
        case 'f':
          // Toggle between first-person and third-person view
          setIsFirstPersonView(prev => {
            const newValue = !prev;
            
            // Immediately update camera position when switching views
            if (newValue && droneRef.current) {
              // Switching TO first-person
              controls.enabled = false;
              const dronePos = droneRef.current.position;
              const offset = firstPersonOffsetRef.current.clone();
              offset.applyQuaternion(droneRef.current.quaternion);
              
              camera.position.set(
                dronePos.x + offset.x,
                dronePos.y + offset.y,
                dronePos.z + offset.z
              );

              const lookDistance = 5;
              const lookDirection = new THREE.Vector3(0, 0, lookDistance);
              lookDirection.applyQuaternion(droneRef.current.quaternion);
              const lookAtPoint = new THREE.Vector3(
                dronePos.x + lookDirection.x,
                dronePos.y + lookDirection.y,
                dronePos.z + lookDirection.z
              );
              camera.lookAt(lookAtPoint);
            } else if (droneRef.current) {
              // Switching TO third-person - position camera behind drone based on its orientation
              const dronePos = droneRef.current.position;
              
              // Calculate camera offset behind and above the drone, considering drone's orientation
              const backwardOffset = new THREE.Vector3(0, 3, -8); // Behind and above
              backwardOffset.applyQuaternion(droneRef.current.quaternion);
              
              camera.position.set(
                dronePos.x + backwardOffset.x,
                dronePos.y + backwardOffset.y,
                dronePos.z + backwardOffset.z
              );
              
              // Point camera at the drone
              controls.target.set(dronePos.x, dronePos.y, dronePos.z);
              camera.lookAt(dronePos);
              controls.enabled = true;
              controls.update();
            }
            
            return newValue;
          });
          break;
        case 'arrowup':
          // Move camera up (only in third-person mode)
          if (!isFirstPersonView) {
            camera.position.y += moveSpeed;
            controls.target.y += moveSpeed;
          }
          break;
        case 'arrowdown':
          // Move camera down (only in third-person mode)
          if (!isFirstPersonView) {
            camera.position.y -= moveSpeed;
            controls.target.y -= moveSpeed;
          }
          break;
        case 'arrowleft':
          // Move camera left (only in third-person mode)
          if (!isFirstPersonView) {
            camera.position.x -= moveSpeed;
            controls.target.x -= moveSpeed;
          }
          break;
        case 'arrowright':
          // Move camera right (only in third-person mode)
          if (!isFirstPersonView) {
            camera.position.x += moveSpeed;
            controls.target.x += moveSpeed;
          }
          break;
      }
      
      if (!isFirstPersonView) {
        controls.update();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isFirstPersonView]);

  // Update drone and obstacles based on frame
  useEffect(() => {
    if (!frame || !droneRef.current) return;

    const sceneRef = droneRef.current.parent as THREE.Scene;
    if (!sceneRef) return;

    // Make drone visible on first frame
    if (!droneRef.current.visible) {
      droneRef.current.visible = true;
    }
    
    // Make target visible on first frame
    if (targetRef.current && !targetRef.current.visible) {
      targetRef.current.visible = true;
    }
    if (targetLightRef.current && !targetLightRef.current.visible) {
      targetLightRef.current.visible = true;
    }

    // Update agent drone position
    const { x, y, z } = frame.agent_position;
    droneRef.current.position.set(x, y, z);
    
    // Update target position from frame data
    if (frame.target_position && targetRef.current && targetLightRef.current) {
      const targetPos = new THREE.Vector3(
        frame.target_position.x,
        frame.target_position.y,
        frame.target_position.z
      );
      targetRef.current.position.copy(targetPos);
      targetLightRef.current.position.copy(targetPos);
    }
    
    // Debug: Log position on frame 0 or every 100 frames
    if (frame.frame === 0 || frame.frame % 100 === 0) {
      console.log(`\n[Frame ${frame.frame}] Agent position: [${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)}]`);
      if (frame.target_position) {
        const tx = frame.target_position.x;
        const ty = frame.target_position.y;
        const tz = frame.target_position.z;
        console.log(`[Frame ${frame.frame}] Target position: [${tx.toFixed(2)}, ${ty.toFixed(2)}, ${tz.toFixed(2)}]`);
        console.log(`  Corridor bounds: X=[${-corridorWidth!/2}, ${corridorWidth!/2}], Y=[0, ${corridorHeight}], Z=[0, ${corridorLength}]`);
        console.log(`  Target inside bounds: X=${tx >= -corridorWidth!/2 && tx <= corridorWidth!/2} (${tx >= -corridorWidth!/2 ? '✓' : '✗'}), Y=${ty >= 0 && ty <= corridorHeight!} (${ty >= 0 && ty <= corridorHeight! ? '✓' : '✗'}), Z=${tz >= 0 && tz <= corridorLength!} (${tz >= 0 && tz <= corridorLength! ? '✓' : '✗'})`);
        if (targetRef.current) {
          const actualPos = targetRef.current.position;
          console.log(`  Target THREE.js position: [${actualPos.x.toFixed(2)}, ${actualPos.y.toFixed(2)}, ${actualPos.z.toFixed(2)}]`);
        }
      }
    }

    // Update drone color based on status
    const topPlate = droneRef.current.children[1] as THREE.Mesh;
    if (topPlate && topPlate.material) {
      const material = topPlate.material as THREE.MeshStandardMaterial;
      if (frame.crashed) {
        material.color.setHex(0xff0000);
        material.emissive.setHex(0xff0000);
      } else if (frame.success) {
        material.color.setHex(0x00ff00);
        material.emissive.setHex(0x00ff00);
      } else {
        material.color.setHex(0x00ffff);
        material.emissive.setHex(0x00ffff);
      }
    }

    // Update or create enemy drones based on obstacles in frame
    if (frame.obstacles && Array.isArray(frame.obstacles)) {
      // Create new enemy drones if count doesn't match
      if (enemiesRef.current.length !== frame.obstacles.length) {
        // Remove old enemies
        enemiesRef.current.forEach(enemy => enemy.dispose(sceneRef));
        enemiesRef.current = [];

        // Create new enemies with diameter and speed from configuration
        frame.obstacles.forEach((obstacle, index) => {
          const diameter = obstacle.diameter || 0.5;
          const speed = obstacle.speed || 3.0;
          const position = new THREE.Vector3(
            obstacle.x || 0,
            obstacle.y || 5,
            obstacle.z || 50
          );
          const enemy = new EnemyDrone(sceneRef, position, diameter, speed, index);
          enemiesRef.current.push(enemy);
        });
      } else {
        // Update existing enemy positions and velocities
        frame.obstacles.forEach((obstacle, index) => {
          if (enemiesRef.current[index]) {
            const position = new THREE.Vector3(
              obstacle.x || 0,
              obstacle.y || 5,
              obstacle.z || 50
            );
            enemiesRef.current[index].setPosition(position);
            
            // Update velocity visualization if velocity data is available
            if (obstacle.vx !== undefined && obstacle.vy !== undefined && obstacle.vz !== undefined) {
              enemiesRef.current[index].updateVelocity({
                x: obstacle.vx,
                y: obstacle.vy,
                z: obstacle.vz
              });
            } else {
              enemiesRef.current[index].updateVelocity(null);
            }
          }
        });
      }
    }

    // Update LIDAR rays visualization
    if (frame.lidar_hit_info && Array.isArray(frame.lidar_hit_info) && frame.lidar_hit_info.length > 0 && droneRef.current) {
      // Remove old rays from drone
      lidarRaysRef.current.forEach(ray => droneRef.current?.remove(ray));
      lidarRaysRef.current = [];
      
      let raysCreated = 0;
      frame.lidar_hit_info.forEach((hit, index) => {
        if (!hit || !hit.direction) return;
        
        // Direction vector is already relative to drone position
        const directionVec = new THREE.Vector3(
          hit.direction[0],
          hit.direction[1],
          hit.direction[2]
        );
        
        const points = [];
        points.push(new THREE.Vector3(0, 0, 0)); // Start from drone center
        points.push(directionVec); // End at direction vector

        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        
        // Green if no obstacle (none/self), Red if hit obstacle (wall or drone)
        let color = 0x00ff00; // Green - clear path
        if (hit.type === 'drone' || hit.type === 'wall') {
          color = 0xff0000; // Red - obstacle detected
        }

        const material = new THREE.LineBasicMaterial({
          color,
          transparent: true,
          opacity: 0.8,
          linewidth: 2,
          depthTest: true,
        });

        const line = new THREE.Line(geometry, material);
        if (droneRef.current) {
          droneRef.current.add(line); // Add to drone so it moves with it
        }
        lidarRaysRef.current.push(line);
        raysCreated++;
      });
      
      // Debug log occasionally
      if (frame.frame % 100 === 0) {
        console.log(`LIDAR: Created ${raysCreated} rays from ${frame.lidar_hit_info.length} hits`);
      }
    } else if (frame.frame % 100 === 0) {
      console.log('LIDAR: No hit info available', frame.lidar_hit_info ? `(${frame.lidar_hit_info.length} items)` : '(undefined)');
    }

    // Update velocity arrow visualization
    if (frame.agent_velocity && droneRef.current) {
      const velocity = new THREE.Vector3(
        frame.agent_velocity.x,
        frame.agent_velocity.y,
        frame.agent_velocity.z
      );
      
      const speed = velocity.length();
      
      // Remove old arrow if it exists
      if (velocityArrowRef.current) {
        droneRef.current.remove(velocityArrowRef.current);
        velocityArrowRef.current.dispose();
        velocityArrowRef.current = null;
      }
      
      // Only show arrow if drone is moving
      if (speed > 0.1) {
        const direction = velocity.clone().normalize();
        const arrowLength = Math.min(speed / (agentMaxSpeed || 10), 1) * agentDiameter * 2; // Scale arrow based on speed percentage
        const arrowColor = 0x00ff00; // Green for velocity
        
        const arrow = new THREE.ArrowHelper(
          direction,
          new THREE.Vector3(0, 0, 0), // Origin at drone center
          arrowLength,
          arrowColor,
          arrowLength * 0.2,
          arrowLength * 0.15
        );
        
        droneRef.current.add(arrow);
        velocityArrowRef.current = arrow;
      }
    }
  }, [frame, agentDiameter, agentMaxSpeed]);

  // Show error if corridor dimensions are missing
  if (!corridorLength || !corridorWidth || !corridorHeight) {
    return (
      <div style={{ position: 'relative' }}>
        <div
          style={{
            width: '100%',
            height: `${height}px`,
            background: '#1a0000',
            borderRadius: '8px',
            overflow: 'hidden',
            border: '2px solid #ff0000',
            boxShadow: '0 0 20px rgba(255, 0, 0, 0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexDirection: 'column',
            gap: '16px',
            padding: '20px'
          }}
        >
          <div style={{ color: '#ff0000', fontSize: '24px', fontWeight: 'bold' }}>
            ⚠ Missing Corridor Dimensions
          </div>
          <div style={{ color: '#ff8888', fontSize: '14px', fontFamily: 'monospace', textAlign: 'center' }}>
            The environment must specify corridor dimensions:<br/>
            Length: {corridorLength ?? 'MISSING'}<br/>
            Width: {corridorWidth ?? 'MISSING'}<br/>
            Height: {corridorHeight ?? 'MISSING'}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ position: 'relative' }}>
      <div
        ref={containerRef}
        style={{
          width: '100%',
          height: `${height}px`,
          background: '#000',
          borderRadius: '8px',
          overflow: 'hidden',
          border: '2px solid #0ff',
          boxShadow: '0 0 20px rgba(0, 255, 255, 0.3)',
        }}
      />
      {/* FPS Counter */}
      <div
        style={{
          position: 'absolute',
          top: 10,
          right: 10,
          background: 'rgba(0, 0, 0, 0.7)',
          border: '1px solid #0ff',
          padding: '5px 10px',
          borderRadius: '4px',
          color: '#0ff',
          fontFamily: 'monospace',
          fontSize: '12px',
        }}
      >
        FPS: {fps}
      </div>
      {/* View Mode Indicator */}
      <div
        style={{
          position: 'absolute',
          top: 10,
          left: 10,
          background: 'rgba(0, 0, 0, 0.7)',
          border: `1px solid ${isFirstPersonView ? '#ff0' : '#0ff'}`,
          padding: '5px 10px',
          borderRadius: '4px',
          color: isFirstPersonView ? '#ff0' : '#0ff',
          fontFamily: 'monospace',
          fontSize: '12px',
          fontWeight: 'bold',
        }}
      >
        {isFirstPersonView ? '👁️ 1st Person' : '🎥 3rd Person'}
      </div>
      {/* Keyboard Controls Help */}
      <div
        style={{
          position: 'absolute',
          bottom: 10,
          left: 10,
          background: 'rgba(0, 0, 0, 0.7)',
          border: '1px solid #0ff',
          padding: '8px 12px',
          borderRadius: '4px',
          color: '#0ff',
          fontFamily: 'monospace',
          fontSize: '11px',
          lineHeight: '1.4',
        }}
      >
        <div style={{ marginBottom: '4px', fontWeight: 'bold', color: '#fff' }}>Controls:</div>
        <div>F - Toggle View</div>
        <div>↑↓←→ - Move Camera</div>
        <div>Mouse - Rotate/Pan/Zoom</div>
      </div>
      {/* Speed Indicator */}
      {frame?.agent_velocity && (() => {
        const velocity = frame.agent_velocity;
        const speed = Math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2);
        const speedPercent = Math.min((speed / (agentMaxSpeed || 10)) * 100, 100);
        return (
          <div
            style={{
              position: 'absolute',
              bottom: 10,
              right: 10,
              background: 'rgba(0, 0, 0, 0.7)',
              border: '1px solid #0ff',
              padding: '8px 12px',
              borderRadius: '4px',
              color: '#0ff',
              fontFamily: 'monospace',
              fontSize: '11px',
              minWidth: '120px',
            }}
          >
            <div style={{ marginBottom: '4px', fontWeight: 'bold', color: '#fff' }}>Agent Speed</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ flex: 1, height: '6px', background: '#222', borderRadius: '3px', overflow: 'hidden' }}>
                <div 
                  style={{ 
                    height: '100%', 
                    width: `${speedPercent}%`,
                    background: speedPercent > 80 ? '#f00' : speedPercent > 50 ? '#ff0' : '#0f0',
                    transition: 'width 0.2s ease',
                  }} 
                />
              </div>
              <div style={{ minWidth: '60px', textAlign: 'right' }}>
                {speed.toFixed(1)} m/s
              </div>
            </div>
            <div style={{ fontSize: '9px', color: '#888', marginTop: '2px' }}>
              Max: {(agentMaxSpeed || 10).toFixed(1)} m/s
            </div>
          </div>
        );
      })()}
      {/* Obstacle/Swarm Info Panel */}
      {frame?.obstacles && frame.obstacles.length > 0 && (
        <div
          style={{
            position: 'absolute',
            top: 40,
            right: 10,
            background: 'rgba(0, 0, 0, 0.7)',
            border: '1px solid #f80',
            padding: '8px 12px',
            borderRadius: '4px',
            color: '#f80',
            fontFamily: 'monospace',
            fontSize: '11px',
            maxWidth: '200px',
          }}
        >
          <div style={{ marginBottom: '4px', fontWeight: 'bold', color: '#fff' }}>
            Swarm Drones: {frame.obstacles.length}
          </div>
          {(() => {
            const sizes = frame.obstacles.map(o => o.diameter).filter(d => d);
            const speeds = frame.obstacles.map(o => o.speed).filter(s => s);
            if (sizes.length > 0 && speeds.length > 0) {
              const avgSize = (sizes.reduce((a, b) => a + b, 0) / sizes.length).toFixed(2);
              const avgSpeed = (speeds.reduce((a, b) => a + b, 0) / speeds.length).toFixed(1);
              const minSize = Math.min(...sizes).toFixed(2);
              const maxSize = Math.max(...sizes).toFixed(2);
              return (
                <>
                  <div style={{ fontSize: '9px', color: '#ffa366', marginTop: '2px' }}>
                    Size: {minSize === maxSize ? `${avgSize}m` : `${minSize}-${maxSize}m`}
                  </div>
                  <div style={{ fontSize: '9px', color: '#ffa366' }}>
                    Speed: ~{avgSpeed} m/s
                  </div>
                </>
              );
            }
            return null;
          })()}
        </div>
      )}
    </div>
  );
}
