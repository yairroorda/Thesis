Potree.scriptPath = new URL("../third_party/potree/build/potree", window.location.href).href;

(function(){
  const container = document.getElementById('potree_render_area');
  const viewer = new Potree.Viewer(container);

  viewer.setEDLEnabled(true);
  viewer.setFOV(60);
  viewer.setPointBudget(5_000_000);
  viewer.setEDLEnabled(false);
  viewer.loadSettingsFromURL();

  viewer.loadGUI(() => {
    viewer.setLanguage('en');
    
    // Automatically open the Appearance and Scene menus
    $("#menu_appearance").next().show();
    $("#menu_scene").next().show();
  });

  // Load the Project scene
  const copcPath = './data/bouwkunde.copc.laz';
  Potree.loadPointCloud(copcPath, 'COPC', (e)=>{
    const pc = e.pointcloud;
    pc.material.size = 1.0;
    pc.material.pointSizeType = Potree.PointSizeType.FIXED;
    pc.material.shape = Potree.PointShape.ROUND;
    viewer.scene.addPointCloud(pc);
    viewer.fitToScreen(0.5);
  });
  

  // Load the Optimal Path
  const routePath = './data/optimal_path.copc.laz';
  Potree.loadPointCloud(routePath, 'COPC', (e)=>{
    const pc = e.pointcloud;
    pc.material.size = 1.0; // Made slightly larger to stand out
    pc.material.pointSizeType = Potree.PointSizeType.ADAPTIVE;
    pc.material.shape = Potree.PointShape.SQUARE;

    const THREE_Color = pc.material.color.constructor;
    
    // Force the shader to color by elevation (Z-axis)
    pc.material.activeAttributeName = "elevation";
    
    // Feed it a gradient that is the same color at every height
    const pathColor = new THREE_Color(1, 1, 1); // White color for the path
    pc.material.gradient = [
        [0.0, pathColor],
        [1.0, pathColor]
    ];
    
    viewer.scene.addPointCloud(pc);
  });

})();