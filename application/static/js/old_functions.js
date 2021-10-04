// Functions no longer used, kept in this file just in case

function drawImage(canvas) {
  let context = canvas.node().getContext('2d');
  let image = context.createImageData(dx, dy);

  for (var y = 0, p = -1; y < dy; ++y) {
    for (var x = 0; x < dx; ++x) {
      var c = d3.rgb(color(heatmap[y][x]));
      image.data[++p] = c.r;
      image.data[++p] = c.g;
      image.data[++p] = c.b;
      image.data[++p] = 255;
    }
  }

  context.scale(0.1, 0.1);
  context.drawImage(image, 0, 0, 2*canvas.width, 2*canvas.height);
}


const parseProjectionData = (data) => {
  let domain = {
      xMin: 9999,
      xMax: -9999,
      yMin: 9999,
      yMax: -9999,
  }

  state.predictionRange = [9999, -9999];
  state.lossRange = [9999, -9999]
  data.forEach(d => {
      d[1] = +d[1];
      d[2] = +d[2];
      
      // Update the domain when necessary
      if (d[1] < domain.xMin) {
        domain.xMin = d[1]
      } else if (d[1] > domain.xMax) {
        domain.xMax = d[1]
      } 
      if (d[2] < domain.yMin) {
        domain.yMin = d[2]
      } else if (d[2] > domain.yMax) {
        domain.yMax = d[2]
      }

      // For loss and prediction confidence when applicable
      if (d.length > 4) {
        d[4] = 1 - (+d[4]);
        d[5] = 1 - Math.log(+d[5]);

        if (d[4] < state.predictionRange[0]) {
          state.predictionRange[0] = d[4];
        } else if (d[4] > state.predictionRange[1]) {
          state.predictionRange[1] = d[4];
        }

        if (d[5] < state.lossRange[0]) {
          state.lossRange[0] = d[5];
        } else if (d[5] > state.lossRange[1]) {
          state.lossRange[1] = d[5];
        }
      }
  });
  return domain
} 


$('#checkpointSlider').on('change', function(){
  let value = parseInt($(this).val());

  // Allow confidence and loss for epoch 1+ 
  if (value === 0) {
    $("#prediction").prop("disabled", true);
    $("#loss").prop("disabled", true);
    $("#label").prop("checked", true);
    state.projectionColor = $("#label").val();
  } else {
    $("#prediction").prop("disabled", false);
    $("#loss").prop("disabled", false);
  }

  state.checkpoint = value;
  let text = (value === 0)? "Pretrained" : `Epoch ${value}`;
  $(this).parent().find('#sliderText').html(text);
  loadData();
  
})