Potree.scriptPath = new URL("../third_party/potree/build/potree", window.location.href).href;

(function(){
  const container = document.getElementById('potree_render_area');
  const viewer = new Potree.Viewer(container);
  const demoSelect = document.getElementById('demo_select');

  const demos = {
    groningen: {
      files: [
        {
          path: './data/facades.copc.laz',
          type: 'COPC',
          apply(pointcloud) {
            pointcloud.material.size = 1.0;
            pointcloud.material.pointSizeType = Potree.PointSizeType.FIXED;
            pointcloud.material.shape = Potree.PointShape.ROUND;
          },
        },
        {
          path: './data/viewshed.copc.laz',
          type: 'COPC',
          apply(pointcloud) {
            pointcloud.material.size = 1.0;
            pointcloud.material.pointSizeType = Potree.PointSizeType.FIXED;
            pointcloud.material.shape = Potree.PointShape.ROUND;
            pointcloud.material.activeAttributeName = 'intensity gradient';
            pointcloud.material.intensityRange = [0, 65535];

            const color = pointcloud.material.color.constructor;
            pointcloud.material.gradient = [
              [0.0, new color(0, 0, 1.0)],
              [0.5, new color(1.0, 1.0, 0.0)],
              [1.0, new color(1.0, 0.0, 0.0)],
            ];
          },
        },
      ],
    },
    delft: {
      files: [
        {
          path: './data/bouwkunde.copc.laz',
          type: 'COPC',
          apply(pointcloud) {
            pointcloud.material.size = 1.0;
            pointcloud.material.pointSizeType = Potree.PointSizeType.FIXED;
            pointcloud.material.shape = Potree.PointShape.ROUND;
          },
        },
        {
          path: './data/optimal_path.copc.laz',
          type: 'COPC',
          apply(pointcloud) {
            pointcloud.material.size = 1.0;
            pointcloud.material.pointSizeType = Potree.PointSizeType.ADAPTIVE;
            pointcloud.material.shape = Potree.PointShape.SQUARE;

            const color = pointcloud.material.color.constructor;
            pointcloud.material.activeAttributeName = 'elevation';
            pointcloud.material.gradient = [
              [0.0, new color(1, 1, 1)],
              [1.0, new color(1, 1, 1)],
            ];
          },
        },
      ],
    },
    ventoux: {
      files: [
        {
          path: '../data/france_ventoux_project/facades.copc.laz',
          type: 'COPC',
          apply(pointcloud) {
            pointcloud.material.size = 1.0;
            pointcloud.material.pointSizeType = Potree.PointSizeType.FIXED;
            pointcloud.material.shape = Potree.PointShape.ROUND;
            pointcloud.material.activeAttributeName = 'classification';
          },
        },
      ],
    },
  };

  const params = new URLSearchParams(window.location.search);
  const demoName = demos[params.get('demo')] ? params.get('demo') : 'groningen';

  if (demoSelect) {
    demoSelect.value = demoName;
    demoSelect.addEventListener('change', () => {
      const next = demoSelect.value;
      const url = new URL(window.location.href);
      url.searchParams.set('demo', next);
      window.location.assign(url.href);
    });
  }

  function loadDemo(name) {
    const demo = demos[name] || demos.groningen;

    for (const item of demo.files) {
      Potree.loadPointCloud(item.path, item.type, (event) => {
        const pointcloud = event.pointcloud;
        item.apply(pointcloud);
        viewer.scene.addPointCloud(pointcloud);
        if (item.path === demo.files[0].path) {
          viewer.fitToScreen(0.5);
        }
      });
    }

    if (demo.onReady) {
      demo.onReady();
    }
  }

  viewer.setEDLEnabled(true);
  viewer.setFOV(60);
  viewer.setPointBudget(5_000_000);
  viewer.setEDLEnabled(false);
  viewer.loadSettingsFromURL();

  viewer.loadGUI(() => {
    viewer.setLanguage('en');
    
    loadDemo(demoName);
  });

})();