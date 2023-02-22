import { scrollContent, highlightContent, checkArrays } from './utils.js';
import { renderInstanceView, renderColor } from './instance-view.js';
import { renderImportanceFromState } from './component-view.js';
import { legend } from './legend.js';

const hoverExample = id => {
  d3.select(`.example#${id}`)
    .attr('r', 10)
    .classed('emphasized', true);
}

const endHover = () => {
  d3.select(`.example.emphasized`)
    .classed('emphasized', false)
    .filter(function() {
      return !this.classList.contains('selected')
    })
    .attr('r', 3)
}

const renderTable = (filteredIDs) => {
  // let filteredIDs = filteredData.map(d => d[0]);
  $.ajax({
      type: 'POST',
      contentType: 'application/json',
      url: '../api/filter_table',
      dataType : 'html',
      data: JSON.stringify(filteredIDs),
      success: function(response){
        $('#contentView').html(response);
     }
  });
}


const resetFilters = (state) => {
  state.discrete = [];
  state.continuous = [];
  state.projectionColor = $("#color-select option:selected").val();
  // Hide dropdown and slider
  $(`#btn-group-categorical-select`).hide();
  $(`#range-slider`).hide();
  $(`#range-value`).hide();

  return state;
}



const toggleModeSelect = () => {
  let newMode = (d3.select('#projectionMode').attr('value') === 'encoder')? 'decoder' : 'encoder';
  let newText = (newMode === 'encoder')? '⇒ Decoder Projections' : '⇐ Encoder Projections'; // TODO: Replace with symbol
  d3.select('#projectionMode')
    .attr('value', newMode)
    .text(newText);
}

export const selectExample = (id, state) => {
  $('#loader').show();
  // let exampleID = (typeof(id) === 'string')? +id.split('-')[1]: id;
  let exampleID = id;
  state.selectedExample = id;
  state['example_id'] = exampleID;

  const server_query = d3.json('../api/eval_one', {
      method: "POST",
      body: JSON.stringify({
        'example_id': exampleID,
        'encoder_head': state.encoderHead,
        'decoder_head': state.decoderHead,
      }),
      headers: {
          "Content-type": "application/json; charset=UTF-8"
      }
  })

  server_query.then(response => {
      // state.encoder_importance = response['attn'];
      // let importance = response['head_importance'];
      // let attn_patten = response['attn_pattern'];

      // state.instance_importance = importance;
      // state.instance_pattern = attn_patten;

      state.encoderAttentions = response['encoder_attentions'];
      state.crossAttentions = response['cross_attentions'];
      state.attributions = response['attributions'];


      $("#instanceAttention").prop("disabled", false);
      state.selectedInput = null;
      state.selectedOutput = null;
      state = renderInstanceView(
        response['input_tokens'],
        response['output_tokens'],
        response['input_saliency'],
        // response['label'],
        // response['loss'],
        '#input-token-container',
        '#output-token-container',
        state,
      );

      state.decoderProjection = response['output_projections'];
      state = renderProjection(response['output_projections'], state.projectionSVG, state.projectionWidth, state.projectionHeight, 'decoder', state);

      // let attentionSVG = d3.select("#attention-svg");


      // state = renderImportanceFromState(attentionSVG, state);
      $('#loader').hide();
  });
  return state;
}



export const selectToken = (id, state) => {
  let tokenIdx = id + 1; // Since hidden states start from the first generated token
  state.selectedOutput = tokenIdx;
  
  // TODO: Redundant with function in $(#interpretation-select) in "index.js"
  if (state.interpretation === 'attention') {
    if (state.selectedOutput !== null && state.decoderAttentions !== null) {
      renderColor(state.crossAttentions[state.selectedOutput], state.decoderAttentions[state.selectedOutput], 'output', state);
    }
  } else if (state.interpretation === 'attribution') {
    state.selectedInput = null;
    if (state.selectedOutput !== null && state.attributions !== null) {
      renderColor(state.attributions[state.selectedOutput]['input'], state.attributions[state.selectedOutput]['output'], 'output', state);
    }
  }
  return state;
}



export const renderProjection = (data, svg, width, height, mode, state) => {
  state.projectionMode = mode;

  // Use decoder order as the default color encoding
  // if (mode === 'decoder') {
  //   state.projectionColor = 'id';
  // } else {
  //   state.projectionColor = state.;
  // }

  let margin = {top: 10, right: 10, bottom: 50, left: 300};
  let innerWidth = width - margin.left - margin.right;
  let innerHeight = height - margin.top - margin.bottom;

  // let loc = state.filtersID.split('-').slice(-1)[0];

  // if (state.canvasID.includes('Right')) {
  //   margin.left = innerWidth / 2 + 250;
  // }

  // if (state.comparisonMode) {
  //   innerWidth = innerWidth / 2;
  // } else {
  //   svg.selectAll(`#axis-g-right`).remove();
  // }

  // console.log(state.predictionRange);
  // let predictionRange = [-9999, 9999];
  // let lossRange = [-9999, 9999];

  // Remove previous drawn axes
  svg.selectAll(`#axis-g`).remove();
  let quadTreeData = data['x'].map((x, i) => [data['ids'][i], x, data['y'][i]])
  // Initialize a quadtree for searching nearby points
  let quadTree = d3.quadtree()
    .x(d => d[1])
    .y(d => d[2])
    .addAll(quadTreeData);

  let canvas = d3.select(`#${state.canvasID}`);

  canvas
    .attr('class', 'scatterplot')
    .attr('width', innerWidth - 1)
    .attr('height', innerHeight - 1)
    .style('width', `${innerWidth - 1}px`)
    .style('height', `${innerHeight - 1}px`)
    .style('transform', `translate(${margin.left + 1}px, ${margin.top + 1}px)`);

  // let legend = legend({
  //   color: d3.scaleSequential([0, 100], d3.interpolateViridis),
  //   title: "Temperature (°F)"
  // })
  // console.log("hello");
  // console.log(legend);
  // Define the scales
  let padding = 10;
  let domain = data['domain'];

  let xScale = d3.scaleLinear().domain(domain).range([padding, innerWidth - padding]);
  let yScale = d3.scaleLinear().domain(domain).range([innerHeight - padding, padding]);

  // Append axes to the canvas
  let axisG = svg.append('g')
    .attr('transform', `translate(${margin.left}, ${margin.top})`)
    .attr('id', `axis-g`);

  let xAxis = d3.axisBottom()
    .scale(xScale)
  let yAxis = d3.axisLeft()
    .scale(yScale);

  let xAxisG = axisG.append("g")
    .attr('class', 'axis')
    .attr('transform', `translate(0, ${innerHeight})`)
    .call(xAxis);

  let yAxisG = axisG.append("g")
    .attr('class', 'axis')
    .call(yAxis);

  // TO DO: Change these to be dynamic
  let xLabel = `UMAP Dimension 1 (${mode})`
  let yLabel = `UMAP Dimension 2 (${mode})`
  // let xLabel = (state.projectionType == 'hidden') ? '' : 'Variability';
  // let yLabel = (state.projectionType == 'hidden') ? 'UMAP Dimension 2' : 'Averge Confidence';

  axisG.append('text')             
    .attr('class', 'axis-label')
    .attr('transform', `translate(${innerWidth / 2}, ${innerHeight + margin.bottom})`)
    .text(xLabel);

  axisG.append('text') 
    .attr('class', 'axis-label')
    .attr('transform', 'rotate(-90)')
    .attr('y', (state.comparisonMode && loc === 'right')? -(innerWidth / 2 + 180) : -margin.left)
    .attr('x', 0 - (innerHeight / 2))
    .attr('dy', margin.left - 40)
    .text(yLabel);


  $('#projectionMode').unbind();
  $('#projectionMode').on('click', function(event) {
    toggleModeSelect();
    // console.log(mo)
    if (mode === 'encoder' && state.decoderProjection !== null) {
      renderProjection(state.decoderProjection, svg, width, height, 'decoder', state);
    } else if (mode === 'decoder') {
      renderProjection(state.responseData, svg, width, height, 'encoder', state);
    }
  })

  // Handlers for interactions with canvas 
  const minDist = 6;
  canvas.on('click', (event) => {
    let mouse = d3.pointer(event)

    let x0 = transform.invertX(mouse[0]);
    let y0 = transform.invertY(mouse[1]);

    let x = xScale.invert(x0);
    let y = yScale.invert(y0);
    let closest = quadTree.find(x, y);

    // console.log(closest, data);
    // // Compute the Euclidean distance with the closest points
    let dx = xScale(closest[1]);
    let dy = yScale(closest[2]);
    let dist = Math.sqrt((dx - x0)**2 + (dy - y0)**2);

    if (dist <= minDist) {
      let index = closest[0];

      // Selection in encoder mode
      if (mode === 'encoder') {

        state.selectedExample = index;
        state = selectExample(index, state); // This function renders decoder hidden states
        $('#projectionMode').show();
        toggleModeSelect();
      } 

      // Selection in decoder mode
      else {
        selectToken(index, state);
        console.log(`selected decoder token ${index + 1}`)
      }
      // scrollContent(index);
      draw(transform);
    }
  });


  // canvas.call(zoomBehaviour);
  let context = canvas.node().getContext('2d');
  let range, index, value;

  const draw = (transform) => {
    const xRescale = transform.rescaleX(xScale);
    const yRescale = transform.rescaleY(yScale);

    // Rescale the axes
    xAxisG.call(xAxis.scale(xRescale));
    yAxisG.call(yAxis.scale(yRescale));
    let colorAttrName = state.projectionColor;
    let colorAttr = null;
    let colorDomain;
    let colorAttrType = 'discrete'; // Discrete by default

    // Discrete attribute
    if (data['discrete'].map(attr => attr.name).includes(colorAttrName)) {
      colorAttr = data['discrete'].find(attr => attr.name === colorAttrName);
      colorDomain = colorAttr.domain;
      // attrDomain = state['discrete'][attrName].domain;
    } 
    // Continuous attribute
    else if (data['continuous'].map(attr => attr.name).includes(colorAttrName)) {
      colorAttr = data['continuous'].find(attr => attr.name === colorAttrName);
      colorDomain = [colorAttr.min, colorAttr.max];
      colorAttrType = 'continuous';
    }

    // else if (colorAttrName == 'id') {
    //   colorAttr = {'values': data['ids']};
    //   colorDomain = [0, data['ids'].length - 1];
    // }

    context.clearRect(0, 0, innerWidth, innerHeight);

    let filteredIDs = [];
    let selectedPoint = null;
    let selectedIdx = (mode === 'encoder')? state.selectedExample : state.selectedOutput - 1;

    data['ids'].forEach((id, i) => {
      context.strokeWidth = 1;
      context.strokeStyle = 'white';
      let cx = xRescale(+data['x'][i]);
      let cy = yRescale(+data['y'][i]);
      let r = (mode == 'encoder')? 3: 6;

      let withinFilterRange = true;
      let attributeSelected = true;

      state.continuous.forEach(attr => {
        if (attr.values[i] < attr.filterRange[0] || attr.values[i] > attr.filterRange[1]) {
          withinFilterRange = false;
        }
      })

      state.discrete.forEach(attr => {
        if (!attr.selected.includes(String(attr.values[i]))) {
          attributeSelected = false;
        }
      })

      if (withinFilterRange && attributeSelected) {
        filteredIDs.push(id);

        
        if (colorAttr !== null) {
          value = +colorAttr.values[i];

          if (colorDomain !== null) {
            value = (value - colorDomain[0]) / (colorDomain[1] - colorDomain[0]);
          }
        } else {
          value = 0;
        }

        // TODO: add color scale for discrete attributes with more than 10 classes
        let fillStyle = (colorAttrType === 'discrete')? d3.schemeSet1[value] : d3.interpolateReds(value); 

        if (selectedIdx === id) {
          selectedPoint = [cx, cy, fillStyle];   
        }

        context.fillStyle = fillStyle
        context.beginPath();
        context.arc(cx, cy, r, 0, 2 * Math.PI);
        context.closePath();
        context.fill();
        context.stroke();
      }

    })

    // Draw the selected point at the end
    if (selectedPoint !== null) {
      context.fillStyle = selectedPoint[2];
      context.strokeWidth = 1;
      context.strokeStyle = 'black';
      context.beginPath()
      context.arc(selectedPoint[0], selectedPoint[1], 10, 0, 2 * Math.PI);
      context.closePath();
      context.fill();
      context.stroke();
    }

    return filteredIDs;
  }

  let transform = d3.zoomIdentity;
  draw(transform);

  let zoomEndTimeout;

  let zoomBehaviour = d3.zoom()
    .scaleExtent([1, 100])
    .on('zoom', (event, d) => {
      clearTimeout(zoomEndTimeout);
      transform = event.transform
      draw(transform);
      context.restore();
    })
  canvas.call(zoomBehaviour);


  // Changing encoded color
  // $('#projectionColor').on('change', function(){
    
  // })

  // Content View
  // $(".content").on('click', function(){
  //   let id = +$(this).attr('id').split('-')[1];
  //   state = selectExample(id, state);
  //   if (state.selectedIdx.has(id)) {
  //     state.selectedIdx.delete(id);
  //   } else {
  //     state.selectedIdx.add(id);
  //   }
  //   highlightContent(id);
  //   draw(transform);
  // }).mouseenter(function() {
  //   let id = `example-${$(this).attr('id').split('-')[1]}`;
  //   hoverExample(id);
  // }).mouseleave(endHover);

  // Initialize sliders based on range
  
  let sliderID = `#range-slider`;
  // $(sliderID).slider("destroy");
  if ($(sliderID).data("uiSlider")){
    $(sliderID).slider("destroy");
    $(`#range-value`).html('0-0');
    $(`#filter-select`).val("none");
  }



  $(sliderID).slider({
    range: true,
    min: 0,
    max: 0,
    step: 0.01,
    values: [0, 0],
    slide: function( event, ui ) {
      $(`#range-value`).html(`${ui.values[0].toFixed(3)}-${ui.values[1].toFixed(3)}`);

      let attrName = $(`#color-select option:selected`).val();
      let attrIndex = state.continuous.findIndex(d => d.name === attrName);
      state.continuous[attrIndex].filterRange = ui.values;
      let filteredIDs = draw(transform);
      // renderTable(filteredIDs);
      // d3.selectAll('.example')
      //   .style('visibility', d => ((1 - d[4]) > state.predictionRange[0] && (1 - d[4]) < state.predictionRange[1])? 'visible' : 'hidden');
      // console.log(state.predictionRange, ui.values)
    }
  });

  $(`#categorical-select`).off('change');
  $(`#categorical-select`).on('change', function(){
    let attrName = $(`#color-select option:selected`).val();
    let attrIndex = state.discrete.findIndex(d => d.name === attrName);

    // There's probably a much cleaner way to do this, look to change later
    let selected = Array.from(document.querySelectorAll(`#btn-group-categorical-select > .vsb-menu > ul.multi > li.active`))
      .map(d => d.getAttribute('value'));

    // Re-draw canvas
    if (!checkArrays(selected, state.discrete[attrIndex].selected)) {
      state.discrete[attrIndex].selected = selected;
      let filteredIDs = draw(transform);
      // renderTable(filteredIDs);

    }

  })

  $("#resetZoom").on('click', () => {
     const t = d3.zoomIdentity.translate(0, 0).scale(1);

     canvas.transition()
       .duration(200)
       .ease(d3.easeLinear)
       .call(zoomBehaviour.transform, t)
  });





  // TODO: The following code into a function
  let selectName = '#projectionColor';
  // let filtersName = `#${state.filtersID}`;

  // $(`${selectName} option:not(:first)`).remove().end();
  $('#color-select option:not(:first)').remove().end();
  state = resetFilters(state);

  data['discrete'].forEach(d => {
    d.selected = d.domain;
    let attrName = d['name'];
    // $(`${selectName}`).append(new Option(attrName, attrName));
    state.discrete.push(d);
    $(`#color-select`).append(new Option(attrName, attrName));
    // state.discreteFilters[attrName] = d.domain;
  })

  data['continuous'].forEach(d => {
    d.filterRange = [d.min, d.max];
    let attrName = d['name'];
    // $(`${selectName}`).append(new Option(attrName, attrName));
    state.continuous.push(d);
    // state.continuousFilters[attrName] = [d.min, d.max];

    // Update range filters 
    $('#color-select').append(new Option(attrName, attrName));

  })
  

  $(`${selectName}`).val(state.projectionColor);
  // console.log($(`${selectName} option[value='${state.projectionColor}']`).length)
  // if ($(`${selectName} option[value='${state.projectionColor}']`).length > 0) {
  // }

  // When a selection is made
  $('#color-select').on('change', function(){
    // let selectedField = $(this).attr('id');
    let selectedValue = $(this).find(":selected").val();
    state.projectionColor = selectedValue;
    draw(transform);
    // let selectedClassList = $(this).attr('class').split(' ');
    // state[selectedField] = selectedValue;
    // let selectedValue = $(this).find(":selected").attr('value');
    // For projection filter selections
    // if (selectedClassList.includes('color-select')){
      // let isComparison = selectedField.includes('right');
      // let loc = (isComparison)? 'right' : 'left';
    let attribute;
      // For categorical attributes
    if (data.discrete.find(attr => attr.name === selectedValue) !== undefined) {
      attribute = data.discrete.find(attr => attr.name === selectedValue);
        
      $(`#categorical-select option`).remove().end();

      // Create option for all values in the domain
      attribute.domain.forEach(val => {
        $(`#categorical-select`).append(new Option(val, val, attribute.selected.includes(val)));
      })
      state.discreteSelect = new vanillaSelectBox(`#categorical-select`, {
        'disableSelectAll': true,
      });

      // Hide slider and show dropdown
      $(`#range-slider`).hide();
      $(`#range-value`).hide();
      // $(`#categorical-select-${loc}`).show();
      $(`#btn-group-categorical-select`).show();

    } 

      // For continuous attributes
    else if (data.continuous.find(attr => attr.name === selectedValue) !== undefined){
      attribute = data.continuous.find(attr => attr.name === selectedValue);

      // Update slider with current selected range
      $(`#range-value`).html(`${attribute.filterRange[0].toFixed(3)}-${attribute.filterRange[1].toFixed(3)}`)
      $(`#range-slider`).slider('option', {
        min: attribute.min,
        max: attribute.max,
        step: (attribute.max - attribute.min) / 100,
        values: attribute.filterRange,
      });

      // Hide dropdown and show slider
      // $(`#categorical-select-${loc}`).hide();
      $(`#btn-group-categorical-select`).hide();
      $(`#range-slider`).show();
      $(`#range-value`).show();

    }

  })






  return state;
}

