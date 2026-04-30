import maplibregl from 'https://esm.sh/maplibre-gl@4.1.2?bundle';
import { LidarControl } from 'https://esm.sh/maplibre-gl-lidar@0.11.1?bundle';

const map = new maplibregl.Map({
    container: 'map',
    style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    center: [4.3729, 52.0065], 
    zoom: 16,
    pitch: 45,
    maxPitch: 85
});

map.on('load', () => {
    // Setup Terrain
    map.addSource('maplibre-terrain-rgb', {
        type: 'raster-dem',
        tiles: ['https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'],
        encoding: 'terrarium',
        tileSize: 256,
        maxzoom: 15
    });
    map.setTerrain({ source: 'maplibre-terrain-rgb', exaggeration: 1 });

    // Initialize LiDAR Control
    const lidarControl = new LidarControl({
        title: "Thesis COPC Viewer",
        colorScheme: "classification",
        pointSize: 1,
        zOffsetEnabled: true,
        autoZoom: true
    });
    map.addControl(lidarControl, 'top-right');

    // Point cloud URL
    const bouwkundeUrl = new URL('./viewer/data/facades.copc.laz', window.location.origin).href;
    // const ventouxUrl = new URL('./viewer/data/ventoux.copc.laz', window.location.origin).href;

    lidarControl.loadPointCloudStreaming(bouwkundeUrl);
    // lidarControl.loadPointCloudStreaming(ventouxUrl);
});