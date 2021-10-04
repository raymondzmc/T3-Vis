const renderColor = (color, isAttention, state) => {

    d3.selectAll('.clickable')
        .style('outline', 'none');

    if (isAttention) {
      d3.selectAll(`.token-${state.inputIdx}`)
        .style("outline", 'thin solid red');
    } else {
      d3.selectAll(`.output-${state.outputIdx}`)
        .style("outline", 'thin solid red');
    }

    // colorRight = attention['BERT-EXT'][selection.Layer - 1][selection.Head - 1][idx];

    let leftMin = Math.min(...color);
    let leftMax = Math.max(...color) + 0.05;
    let bg_color;

    color.forEach((value, i) => {
        value = +value;
        if (value === 0) {
          bg_color = '#eee';
        } else {
          if (isAttention) {
            bg_color = d3.interpolateReds((value - 0) / 100);
          } else {
            bg_color = d3.interpolateReds(value);
          }
          
        }
        d3.select(`#token-${i}`)
            .style('background-color', bg_color);
    })
}

export const renderInstanceView = (tokens, output, inputSaliency, label, loss, inputContainer, outputContainer, state) => {

  let inputs = d3.select(inputContainer).selectAll('li.input-token').data(tokens);
  let outputs = d3.select(outputContainer).selectAll('li.output-logit').data(output);

  let inputsEnter = inputs.enter().append('li');
  let outputsEnter = outputs.enter().append('li');
  d3.select('#input-container-title').text(`Input Tokens:`)
  d3.select('#output-container-title').text(`Output Logits:`)
  let metadataList = d3.select('#instance-metadata-list');
  
  metadataList.selectAll('li').remove()
  metadataList.append('li')
    .classed('list-group-item', true)
    .text(`Ground Truth Label: ${label}`);

  metadataList.append('li')
    .classed('list-group-item', true)
    .text(`Loss: ${Math.round(loss * 1000) / 1000}`);

  inputsEnter.merge(inputs)
      .transition(300)
      .text(d => d)
      .attr('id', (d, i) => `token-${i}`)
      .attr('class', (d, i) => `input-token token-${i} clickable`)
      .style('background-color', '#eee');

  outputsEnter.merge(outputs)
      .transition(300)
      .text(d => `${Math.round(d * 1000) / 1000}`)
      .attr('id', (d, i) => `output-${i}`)
      .attr('class', (d, i) => `output-logit output-${i} clickable`)
      .style('background-color', '#eee');


  // let width = $("#modal").width();
  // let height = $("#modal").height() / 2;
  // let margin = {'top': 20, 'left': 0, 'right': 0, 'bottom': 50}

  d3.select('svg.svg-container').remove();

  d3.selectAll('.input-token').style('outline', 'none');

  $(function(){
    $(".input-token").on("click", function() {
      console.log(state.interpMethod);
      if (state.interpMethod === 'attention') {

        let layer = state.attentionIdx[0];
        let head = state.attentionIdx[1];
        if (layer !== null && head !== null){
          let idx = +$(this).attr('id').split('-')[1];
          state.inputIdx = idx;
          let color = state.attention[layer - 1][head - 1][idx];
          renderColor(color, true, state);
        }
      }
    });
  });

  $(function(){
    $(".output-logit").on("click", function() {
      if (state.interpMethod !== 'attention') {
        let outputIdx = d3.select(this).attr('id').split('-')[1];
        let color = inputSaliency[state.interpMethod][outputIdx];
        state.outputIdx = outputIdx;
        renderColor(color, false, state);
      }
    });
  });

  inputs.exit().remove();
  outputs.exit().remove();
  return state;
}