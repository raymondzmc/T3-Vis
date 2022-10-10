export const renderColor = (inputColor, outputColor, selected, state) => {

  d3.selectAll('.clickable')
      .style('outline', 'none')
      .style('background-color', '#eee');

  if (selected === 'input') {
    d3.select(`#input-token-${state.selectedInput}`)
      .style("outline", 'thin solid red');
  } else {
    d3.select(`#output-token-${state.selectedOutput}`)
      .style("outline", 'thin solid red');
  }
  // colorRight = attention['BERT-EXT'][selection.Layer - 1][selection.Head - 1][idx];

  // let leftMin = Math.min(...color);
  // let leftMax = Math.max(...color) + 0.05;
  let bg_color;

  inputColor.forEach((value, i) => {
      value = +value;
      if (value === 0) {
        bg_color = '#eee';
      } else {
        bg_color = d3.interpolateReds(value / 100);
      }
      d3.select(`#input-token-${i}`)
          .style('background-color', bg_color);
  })


  outputColor.forEach((value, i) => {
      value = +value;
      if (value === 0) {
        bg_color = '#eee';
      } else {
        bg_color = d3.interpolateReds(value / 100);
      }
      d3.select(`#output-token-${i}`)
          .style('background-color', bg_color);
  })
}

export const renderInstanceView = (tokens, output, inputSaliency, inputContainer, outputContainer, state) => {

  let inputs = d3.select(inputContainer).selectAll('li.input-token').data(tokens);
  let outputs = d3.select(outputContainer).selectAll('li.output-logit').data(output);

  let inputsEnter = inputs.enter().append('li');
  let outputsEnter = outputs.enter().append('li');
  d3.select('#input-container-title').text(`Input Tokens:`)
  d3.select('#output-container-title').text(`Output Tokens:`)
  let metadataList = d3.select('#instance-metadata-list');
  
  metadataList.selectAll('li').remove()
  // metadataList.append('li')
  //   .classed('list-group-item', true)
  //   .text(`Ground Truth Label: ${label}`);

  // metadataList.append('li')
  //   .classed('list-group-item', true)
  //   .text(`Loss: ${Math.round(loss * 1000) / 1000}`);

  inputsEnter.merge(inputs)
    .transition(300)
    .text(d => d)
    .attr('id', (d, i) => `input-token-${i}`)
    .attr('class', (d, i) => `input-token clickable`)
    .style('background-color', '#eee');

  
      

  outputsEnter.merge(outputs)
      .transition(300)
      .text(d => d)
      .attr('id', (d, i) => `output-token-${i}`)
      .attr('class', (d, i) => `output-token clickable`)
      .style('background-color', '#eee');


  // let width = $("#modal").width();
  // let height = $("#modal").height() / 2;
  // let margin = {'top': 20, 'left': 0, 'right': 0, 'bottom': 50}

  d3.select('svg.svg-container').remove();

  d3.selectAll('.clickable')
      .style('outline', 'none')
      .style('background-color', '#eee');

  inputsEnter.merge(inputs)
    .on('click', function() {
      state.selectedInput = d3.select(this).attr('id').split('-')[2];
      if (state.encoderHead != null) {
        let color = state.encoderAttentions[state.selectedInput];
        renderColor(color, [], 'input', state);
      }
    });

  outputsEnter.merge(outputs)
    .on('click', function() {
      state.selectedOutput = d3.select(this).attr('id').split('-')[2];
      if (state.decoderHead != null) {
        let inputColor = state.crossAttentions[state.selectedOutput];
        let outputColor = state.decoderAttentions[state.selectedOutput];
        renderColor(inputColor, outputColor, 'output', state);
      }
    });

  // $(".input-token").on("click", function() {
  //   console.log(state.interpMethod);
  //   if (state.interpMethod === 'attention') {

  //     // let layer = state.attentionIdx[0];
  //     // let head = state.attentionIdx[1];
  //     // if (layer !== null && head !== null){
  //       let idx = +$(this).attr('id').split('-')[1];
  //       state.selectedInput = idx;
  //       let color = state.encoderAttentions[idx];
  //       renderColor(color, true, state);
  //     // }
  //   }
  // });

  // $(function(){
  //   $(".output-logit").on("click", function() {
  //     if (state.interpMethod !== 'attention') {
  //       let outputIdx = d3.select(this).attr('id').split('-')[1];
  //       let color = inputSaliency[state.interpMethod][outputIdx];
  //       state.selectedOutput = idx;
  //       renderColor(color, false, state);
  //     }
  //   });
  // });

  inputs.exit().remove();
  outputs.exit().remove();
  return state;
}