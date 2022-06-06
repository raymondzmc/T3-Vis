import { renderProjection } from './projection-view.js';
import { renderImportanceFromState, renderImportance } from './component-view.js'
import { selectExample } from './utils.js'

let state = {
  'corpus': 'sst_demo',
  'comparisonMode': false,
  'projectionType': 'hidden',
  'selectedIdx': new Set(),
    
  'type': 'train',
  'model': 'bert',
  'hiddenLayer': 12,

  'discrete': [],
  'continuous': [],

  'discreteFilters': {},
  'continuousFilters': {},
  
  'projectionColor': $("#projectionColor option:selected").val(),
  'canvasID': 'canvasLeft',
  'filtersID': 'filters-left',
  // 'predictionRange': [99999, -99999],
  // 'lossRange': [99999, -99999],
  // 'classFilter': {
  //   0: true,
  //   1: true,
  // },

  'checkpointName': $("#checkpointName option:selected").val(),

  'attentionIdx': [null, null],
  'inputIdx': null,
  'outputIdx': null,
  'attention': null,
  'attentionView': $('#attentionViewSelect').find(':checked').val(),
  'attentionScale': $('#attentionScaleSelect').find(':checked').val(),
  'aggregate_importance': null,
  'instance_importance': null,
  'aggregate_pattern': null,
  'instance_pattern': null,

  
  'interpMethod': $('#interpretationSelect').find(':checked').val(),

  'pruned_heads': {},

  'comparisonState': null,
}

$('#projectionTabs > li > a').click(function() {
  if (state.projectionType !== $(this).attr('value')) {
    state.projectionType = $(this).attr('value');

    if (state.projectionType === 'hidden') {
      $('#hiddenLayer').removeClass('disabled');
    } else {

      // if (state.checkpointName <= 1) {
      //   $('#checkpointSlider').val(2).change();
      // }

      $('#hiddenLayer').addClass('disabled');
    }

    state = loadData(state);
  }
});


// Attention View
$("#attentionViewSelect").change(function(){
  let value = $(this).find(':checked').val();
  state.attentionView = value;
  if (value === 'importance') {
    d3.selectAll('.attn-pattern').attr('visibility', 'hidden');
  } else {
    d3.selectAll('.attn-pattern').attr('visibility', 'visible');
  }
});


// $("#attentionScaleSelect").change(function(){
//   let value = $(this).find(':checked').val();
//   state.attentionScale = value;
//   state = renderImportanceFromState(attentionSVG, state);
// });

$('#attention-comparison-select').change(function(){
  let value = $(this).find(':checked').val();
  if (value == 'right') {
    state.comparisonState = renderImportanceFromState(attentionSVG, state.comparisonState);
  } else {
    state = renderImportanceFromState(attentionSVG, state.comparisonState);
  }
})


$("#comparison-mode").change(function(){
  let value = $(this).find(':checked').val();
  let compare = (value === 'on')? true : false;
  state.comparisonMode = compare;
  
  if (compare) {
    $('#attention-right').prop("disabled", false);
    $('#interpretation-right').prop("disabled", false);
    $('#projection-options').hide();
    $('#projection-options-compare').show();
    state.comparisonState = Object.assign({}, state);
    state.comparisonState.canvasID = state.canvasID.replace('Left', 'Right');
    state.comparisonState.filtersID = state.filtersID.replace('left', 'right');

    $(`#${state.comparisonState.canvasID}`).show();
    $(`#${state.comparisonState.filtersID}`).show();

    state.comparisonState = loadData(state.comparisonState);

  } else {
    $('#attention-right').prop("disabled", true);
    $('#interpretation-right').prop("disabled", true);
    $('#projection-options').show();
    $('#projection-options-compare').hide();
    $(`#${state.comparisonState.canvasID}`).hide();
    $(`#${state.comparisonState.filtersID}`).hide();
    state.comparisonState = null;
  }

  state = loadData(state);
  
});



$('#reset-heads').on('click', function(event) {
  state.pruned_heads = {};

  if (state.selectedIdx.size !== 0) {
    let it = state.selectedIdx.values()
    let idx = it.next().value
    state = selectExample(idx, state);
    state = renderImportanceFromState(attentionSVG, state);
  } else {
    state = renderImportanceFromState(attentionSVG, state);
  }

});

$('#compute-instance').on('click', function(event) {
  // console.log(state.selectedIdx);
  console.log(state['example_id']);
  if (state['example_id'] !== undefined) {
    // let it = state.selectedIdx.values()
    // let idx = it.next().value
    // console.log(idx)
    state = selectExample(state['example_id'], state);
  } else {
    state = renderImportanceFromState(attentionSVG, state);
  }

});


// Instance Investigation View
$("#interpretationSelect").change(function(){
  let value = $(this).find(':checked').val();
  state.interpMethod = value;
});


// Projection View
$('select').on('change', function(){
  let selectedField = $(this).attr('id');
  let selectedValue = $(this).find(":selected").val();
  let selectedClassList = $(this).attr('class').split(' ');

  // For projection filter selections
  if (selectedClassList.includes('filter-select')){
    let isComparison = selectedField.includes('right');
    let loc = (isComparison)? 'right' : 'left';
    let attribute;

    // For categorical attributes
    if (state.discrete.find(attr => attr.name === selectedValue) !== undefined){
      attribute = (isComparison)?
        state.comparisonState.discrete.find(attr => attr.name === selectedValue) :
        state.discrete.find(attr => attr.name === selectedValue);


      $(`#categorical-select-${loc} option`).remove().end();

      // Create option for all values in the domain
      attribute.domain.forEach(val => {
        $(`#categorical-select-${loc}`).append(new Option(val, val, attribute.selected.includes(val)));
      })
      state.discreteSelect = new vanillaSelectBox(`#categorical-select-${loc}`, {
        'disableSelectAll': true,
      });

      // Hide slider and show dropdown
      $(`#range-slider-${loc}`).hide();
      $(`#range-value-${loc}`).hide();
      // $(`#categorical-select-${loc}`).show();
      $(`#btn-group-categorical-select-${loc}`).show();

    } 

    // For continuous attributes
    else {
      attribute = (isComparison)?
        state.comparisonState.continuous.find(attr => attr.name === selectedValue) :
        state.continuous.find(attr => attr.name === selectedValue);

      // Update slider with current selected range
      $(`#range-value-${loc}`).html(`${attribute.filterRange[0].toFixed(3)}-${attribute.filterRange[1].toFixed(3)}`)
      $(`#range-slider-${loc}`).slider('option', {
        min: attribute.min,
        max: attribute.max,
        step: (attribute.max - attribute.min) / 100,
        values: attribute.filterRange,
      });

      // Hide dropdown and show slider
      // $(`#categorical-select-${loc}`).hide();
      $(`#btn-group-categorical-select-${loc}`).hide();
      $(`#range-slider-${loc}`).show();
      $(`#range-value-${loc}`).show();

    }

  }

  // For categorical value selections
  else if (selectedClassList.includes('categorical-select')){
    
    // This is handled in the projection view to avoid rendering entire projection

  } 

  // For other selections
  else {


    // if (state.type !== 'train') {
    //   $('#cartographyTab').addClass('disabled');
    // } else {
    //   $('#cartographyTab').removeClass('disabled');
    // }

    // TODO: Probably not the best way to do this right now, change this to be more robust
    if (state.comparisonMode) {


      if (selectedField.includes('Left')){
        selectedField = selectedField.replace('Left', '');
        state[selectedField] = selectedValue;
        state = loadData(state);
      } else if (selectedField.includes('Right')) {
        selectedField = selectedField.replace('Right', '');
        state.comparisonState[selectedField] = selectedValue;
        state.comparisonState = loadData(state.comparisonState);
      } else {
        state[selectedField] = selectedValue;
      }



    } else {
      state[selectedField] = selectedValue;

      if (selectedField === 'hiddenLayer' || selectedField === 'checkpointName') {
        state = loadData(state);
      }

    }
  }
  // if (state.projectionType === 'hidden') {
  //   d3.select('#projectionView').style('height', '360%');
  // } else {
  //   d3.select('#projectionView').style('height', '600%');
  // }

})


const projectionWidth = 1200;
const projectionHeight = 500;
const projectionSVG = d3.select("#projectionView")
  .append('svg')
  .attr("class", "scatterplot")
  .attr("width", projectionWidth)
  .attr("height", projectionHeight);

const projectionCanvasLeft = d3.select("#projectionView")
  .append('canvas')
    .attr('id', state.canvasID);

const projectionCanvasRight = d3.select("#projectionView")
  .append('canvas')
    .attr('id', state.canvasID.replace('Left', 'Right'));

const attentionSVG = d3.select("#attentionView")
  .append('svg')
    .attr("class", "projection")
    .attr("id", "attention-svg")
    // .attr("width", 700)
    .attr("height", 500);



// const tokenList = d3.select("#instanceView")
  // .append("ul")
  // .attr("id", "tokens")
  // .attr('class', 'token-container')
  // .attr("width", 1200)
  // .attr("height", 500);

const resetFilters = (state) => {
  state.discrete = [];
  state.continuous = [];

  let loc = state.filtersID.split('-').slice(-1)[0];

  // Hide dropdown and slider
  $(`#btn-group-categorical-select-${loc}`).hide();
  $(`#range-slider-${loc}`).hide();
  $(`#range-value-${loc}`).hide();

  return state;
}


const loadData = (state) => {

  $('#loader').show();

  const server_query = d3.json('../api/data', {
      method: "POST",
      body: JSON.stringify(state),
      headers: {
          "Content-type": "application/json; charset=UTF-8"
      }
  })

  server_query.then(response => {
    let importance = response['head_importance'];
    let attn_patten = response['aggregate_attn'];
    console.log(response)
    // attn_patten.forEach(d => {
    //   d.attn = JSON.parse(d.attn);
    // })

    // TODO: Do something to handle the error here
    if (response['x'] === undefined || response['y'] === undefined) {
      $('#loader').hide();
      return;
    }

    state = renderImportance(importance, attn_patten, attentionSVG, 700, 500, state);

    state.aggregate_importance = importance;
    state.aggregate_pattern = attn_patten;


    // TODO: The following code into a function
    let selectName;
    let filtersName = `#${state.filtersID}`;

    if (state.comparisonMode) {
      selectName = (state.canvasID.includes('Right'))? '#projectionColorRight':'#projectionColorLeft';
    } else {
      selectName = '#projectionColor';
    }

    $(`${selectName} option:not(:first)`).remove().end();
    $(`${filtersName} .filter-select option:not(:first)`).remove().end();

    state = resetFilters(state);

    response['discrete'].forEach(d => {
      d.selected = d.domain;
      let attrName = d['name'];
      $(`${selectName}`).append(new Option(attrName, attrName));
      state.discrete.push(d);
      $(`${filtersName} .filter-select`).append(new Option(attrName, attrName));
      // state.discreteFilters[attrName] = d.domain;
    })

    response['continuous'].forEach(d => {
      d.filterRange = [d.min, d.max];
      let attrName = d['name'];
      $(`${selectName}`).append(new Option(attrName, attrName));
      state.continuous.push(d);
      // state.continuousFilters[attrName] = [d.min, d.max];

      // Update range filters 
      $(`${filtersName} .filter-select`).append(new Option(attrName, attrName));

    })

    if ($(`${selectName} option[value='${state.projectionColor}']`).length > 0) {
      $(`${selectName}`).val(state.projectionColor);
    }


    state = renderProjection(response, projectionSVG, projectionWidth, projectionHeight, state);
    $('#loader').hide();

  });
  return state
}





state = loadData(state);


