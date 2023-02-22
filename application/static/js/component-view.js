import { renderColor } from './instance-view.js';

export const renderImportanceFromState = (attentionSVG, state) => {
  let value = state.attentionScale;
  if (value === 'aggregate') {
    state = renderImportance(state.aggregate_importance, state.aggregate_pattern, attentionSVG, 700, 500, state);
  } else {
    state = renderImportance(state.instance_importance, state.instance_pattern, attentionSVG, 700, 500, state);
  }
  return state;
}


export const getAttentionMaps = (attentionType, layer, head, state) => {
  
}


export const renderImportance = (data, attn_patten, svg, width, height, state) => {

  // d3.select('#attentionView').style('position', 'relative');
  // svg.style("position", "absolute")
  // let attn_len = (attn_patten[0].attn.length / 4)**(1/2)
  let headType = svg.attr("value");

  let decoderOnly = true;
  if (decoderOnly) {
    headType = 'encoder';
  }

  let marginX = 50;
  let marginY = 20;
  let nLayers = data.length;
  let nHeads = data[0].length;
  // let residualX = 

  // width = svg.attr('width');
  // width = 700;
  // height = svg.attr('height');
  let innerWidth = width - 2 * marginX;
  let innerHeight = height - 2 * marginY;
  // console.log(svg.attr('width'), width);
  // console.log()
  const xScale = d3
    .scaleBand()
    .domain(Array.from({length: data[0].length}, (_, i) => i + 1))
    .range([marginX, marginX + innerWidth])
    .padding(0.05);

  const yScale = d3
    .scaleBand()
    .domain(Array.from({length: data.length}, (_, i) => i + 1))
    .range([marginY + innerHeight, marginY])
    .padding(0.05);

  // Axis
  svg.selectAll('g').remove()
  let attnPattern = svg.append('g')
    .attr('id', `${headType}-attn-pattern`);

  svg = svg.append('g')
    .attr('id', `${headType}-heatmap`);


  svg.append('g')
    .attr('class', 'x-axis')
    .attr('transform', `translate(0, ${yScale.range()[0]})`)
    .style('font-size', 15)
    .call(d3.axisBottom(xScale).tickSize(0))
    .select(".domain").remove()

  svg.append('g')
    .attr('class', 'y-axis')
    .attr('transform', `translate(${marginX}, 0)`)
    .style('font-size', 15)
    .call(d3.axisLeft(yScale).tickSize(0))
    .select(".domain").remove()

  // let attn_image = d3.select('#projection_4 > svg').append('svg:image');

  // Add classes to ticks
  svg.selectAll('.x-axis > .tick')
    .attr('class', d => `tick head-${d}`);
  svg.selectAll('.y-axis > .tick')
    .attr('class', d => `tick layer-${d}`);

  let rows = svg
    .selectAll('.row')
    .data(data);

  let rowsEnter = rows.enter().append('g').attr('class', 'importance-row');

  rowsEnter
    .merge(rows)
    .attr('transform', (d, i) => `translate(0, ${yScale(i + 1)})`)
    .attr('class', (d, i) => `row-${i + 1}`)
    .classed('row', 'true');

  let cells = rowsEnter.merge(rows)
    .selectAll('.cell')
    .data(d => d);

  let cellsEnter = cells.enter()
    .append('g')
      .attr('class', (d, i) => `col-${i + 1}`)
      .classed('cell', true)
      .attr('transform', (d, i) => `translate(${xScale(i + 1)}, 0)`);

  cellsEnter.merge(cells)
    .on('click', function() {
      let head = d3.select(this).attr('class').match(/(\d+)/)[0];
      let layer = d3.select(this.parentNode).attr('class').match(/(\d+)/)[0];
      state[`${headType}Head`] = [layer, head];

      // Unselect previous selections 
      svg.selectAll('.cell > .selected').classed('selected', false);
      svg.selectAll('.x-axis > .tick.selected').classed('selected', false);
      svg.selectAll('.y-axis > .tick.selected').classed('selected', false);

      // Select heatmap cell and tick on axis
      d3.select(this).classed('selected', true);
      d3.select(this).select('.border').classed('selected', true);
      d3.select(this).select('.prune-button').classed('selected', true);
      svg.select(`.x-axis > .tick.head-${head}`).classed('selected', true);
      svg.select(`.y-axis > .tick.layer-${layer}`).classed('selected', true);

      // Visualize dataset aggregate attentions for the selected head 
      const agg_attn_query = d3.json('../api/agg_attentions', {
        method: "POST",
        body: JSON.stringify({
          'attention_type': headType,
          'layer': layer,
          'head': head,
        }),
        headers: {
          "Content-type": "application/json; charset=UTF-8"
        }
      })
      agg_attn_query.then(response => {
        console.log(response['attn'])
        d3.select(`#${headType}-heatmap`).style("visibility", 'hidden');

        if (headType === 'encoder'){
          drawImage(response);
        } else {
          drawImage2(response);
        }

      })


      // Render the token-level heatmap if "attention" is selected
      if (state.selectedExample !== null) {

        $('#loader').show();
        const server_query = d3.json('../api/attentions', {
          method: "POST",
          body: JSON.stringify({
            'attention_type': headType,
            'layer': layer,
            'head': head,
          }),
          headers: {
            "Content-type": "application/json; charset=UTF-8"
          }
        })

        // TODO: Don't send requests if an example havent't been selected "if (state.selectedInput != null){} "
        server_query.then(response => {
          // Update state with attention maps
          if (headType === 'encoder'){
            state.encoderAttentions = response['encoder_attentions'];
            if (state.selectedInput !== null && state.interpretation === 'attention') {
              renderColor(state.encoderAttentions[state.selectedInput], [], 'input', state);
            }
          } 

          else if (headType === 'decoder') {
            state.crossAttentions = response['cross_attentions'];
            state.decoderAttentions = response['decoder_attentions'];
            if (state.selectedOutput !== null && state.interpretation === 'attention'){
              renderColor(state.crossAttentions[state.selectedOutput], state.decoderAttentions[state.selectedOutput], 'output', state);
            }
          }

          $('#loader').hide();
        });

      }



    }).on('mouseover', function() {
      d3.select(this).select('.prune-button').classed('hovered', true);
      d3.select(this).select('.border').classed('hovered', true);
    }).on('mouseleave', function() {
      d3.select(this).select('.prune-button').classed('hovered', false);
      d3.select(this).select('.border').classed('hovered', false);
    })

  if (headType === 'encoder') {
    let rect = cellsEnter.append('rect');
    // .merge(cells.selectAll('rect')); 
    let rectEnter = rect.enter();

    rectEnter.merge(rect)
      // .attr('class', (d, i) => `head-${i + 1}`)
      .attr('height', yScale.bandwidth())
      .attr('width', xScale.bandwidth())
      .attr('ry', yScale.bandwidth() * 0.1)
      .attr('rx', xScale.bandwidth() * 0.1)
      .attr('fill', d => d3.interpolateReds(d));
      // .on('click', clickHead)

    cellsEnter.append('text').merge(cells.selectAll('text'))
      .attr('dx', `${xScale.bandwidth() / 4}`)
      .attr('dy', `${yScale.bandwidth() / 2}`)
      .style('font-size', `${yScale.bandwidth() / 3.5}px`)
      .text(d => d.toFixed(2));

  } else {
    let rectTop = cellsEnter.append('rect')
      .attr('class', 'rect-top');
    let rectTopEnter = rectTop.enter();
    rectTopEnter.merge(rectTop)
      .attr('height', yScale.bandwidth() / 2)
      .attr('width', xScale.bandwidth())
      .attr('ry', yScale.bandwidth() * 0.1)
      .attr('rx', xScale.bandwidth() * 0.1)
      .attr('fill', d => d3.interpolateBlues(d[0]));
    let textTop = cellsEnter.append('text')
      .attr('class', 'text-top');
    textTop.merge(cells.selectAll('text.text-top'))
      .attr('dx', `${xScale.bandwidth() / 4}`)
      .attr('dy', `${yScale.bandwidth() / 4 + yScale.bandwidth() * 0.1}`)
      .style('font-size', `${yScale.bandwidth() / 3.5}px`)
      .text(d => d[1].toFixed(2));

    let rectBottom = cellsEnter.append('rect')
      .attr('class', 'rect-bottom');
    let rectBottomEnter = rectBottom.enter();
    rectBottomEnter.merge(rectBottom)
      .attr('height', yScale.bandwidth() / 2)
      .attr('width', xScale.bandwidth())
      .attr('ry', yScale.bandwidth() * 0.1)
      .attr('rx', xScale.bandwidth() * 0.1)
      .attr('y', yScale.bandwidth() / 2)
      .attr('fill', d => d3.interpolateReds(d[1]));
    let textBottom = cellsEnter.append('text')
      .attr('class', 'text-bottom');
    textBottom.merge(cells.selectAll('text.text-bottom'))
      .attr('dx', `${xScale.bandwidth() / 4}`)
      .attr('dy', `${yScale.bandwidth() - yScale.bandwidth() * 0.1}`)
      .style('font-size', `${yScale.bandwidth() / 3.5}px`)
      .text(d => d[0].toFixed(2));
    // cellsEnter.append('text').merge(cells.selectAll('text'))
    //   .attr('dx', `${xScale.bandwidth() / 4}`)
    //   .attr('dy', `${yScale.bandwidth() / 2}`)
    //   .style('font-size', `${yScale.bandwidth() / 3.5}px`)
    //   .text(d => d[1].toFixed(2));



    // .merge(cells.selectAll('rect'));
    // let rectEnter = rect.enter();

  }

  // console.time('createRandom');
  // let testAttention = [];
  // for (let h = 0; h < 144; h++) {
  //   testAttention.push([]);
  //   for (let i = 0; i < 512*512; i++) {
  //     testAttention[h].push(Math.floor(Math.random() * 255))
  //   }
  // }
  // console.timeEnd('createRandom');

  // console.log(testAttention);
  const drawImage = async (attn) => {
    attn = attn.attn;

    // if (attentionType === 'encoder'){
      
    // } else if (attentionType === 'decoder'){
    //   decoder_attn = attn.attn[0]
    //   cross_attn = attn.attn[1];
    // }

    // let layer = attn.layer + 1;
    // let head = attn.head + 1;
    // let cell = svg.select(`.row.row-${layer}`).select(`.cell.col-${head}`);
    let canvas;


    // let width = innerWidth;
    // let height = innerHeight;

    // let attn_len = parseInt(Math.sqrt(attn.length));

    let attn_len = (attn.length / 4)**(1/2)
    // console.log(attn_len, typeof(attn_len))

    if (attnPattern.select('canvas').size() === 0) {
      let foreignObject = attnPattern.append('foreignObject')
        .attr('width', width)
        .attr('height', height)
        .attr('class', 'attn-pattern')
        .attr('visibility', 'visible');

      let foBody = foreignObject.append('xhtml:body')
        .attr('width', width)
        .attr('height', height)
        .style('background-color', 'none');

      canvas = foBody.append('canvas')
        .attr('width', width)
        .attr('height', height)
        .style('border', '0.1px solid black');
    } else {
      canvas = attnPattern.select('canvas');
    }
    // console.log();
    // if (cell.select('foreignObject').length())

    
        // .append("xhtml:body");

    let context = canvas.node().getContext('2d');

    // let layer = Math.floor(idx / 12);
    // let head = idx % 12;

    // let context = canvas.node().getContext('2d');

    let imageData = new Uint8ClampedArray;
    imageData = Uint8ClampedArray.from(attn);
    console.log(imageData);
    let image = context.createImageData(attn_len, attn_len);
    image.data.set(imageData);
    let bitmapOptions = {
      'resizeWidth': Math.round(width),
      'resizeHeight': Math.round(height),
    }

    let resizedImage = await window.createImageBitmap(image, 0, 0, image.width, image.height, bitmapOptions);

    context.clearRect(0, 0, height, height);
    context.drawImage(resizedImage, 0, 0);

    attnPattern.append('text')
      .attr('class', 'fas prune-button fa-4x')
      .text('\uf057')
      .attr('x', marginX / 2 + innerWidth)
      .attr('y',  3 * marginY)
      .raise()
      .on('click', function(){
        attnPattern.selectAll('*').remove();
        d3.select(`#${headType}-heatmap`).style("visibility", 'visible');
      });

    return state;
  }


  const drawImage2 = async (attn) => {

    let decoder_attn = attn.attn[0];
    let cross_attn = attn.attn[1];
    let inputLen = Math.round(attn.input_len);
    let outputLen = Math.round(attn.output_len);

    // let decoderAttnGroup = d3.select(`#${headType}-heatmap`).append('g');
    // let crossAttnGroup = d3.select(`#${headType}-heatmap`).append('g');
    //   .attr('transform', `translate(0, ${height})`);



    let crossAttnHeight, crossAttnWidth, decoderAttnWidth, decoderAttnHeight;

    crossAttnHeight = height / 2;
    crossAttnWidth = inputLen / outputLen * crossAttnHeight;
    decoderAttnWidth = crossAttnWidth;
    decoderAttnHeight = decoderAttnWidth;

    let crossAttnMargin = (width - crossAttnWidth) / 2;
    let decoderAttnMargin = (width - decoderAttnWidth) / 2;
    console.log(crossAttnMargin, decoderAttnMargin);
    // if (outputLen >= inputLen){
      
    // } else {
    //   crossAttnWidth = width;
    //   crossAttnHeight = outputLen / inputLen * (height  / 2);
    //   decoderAttnWidth = crossAttnWidth;
    //   decoderAttnHeight = decoderAttnWidth;
    // }

    // let layer = attn.layer + 1;
    // let head = attn.head + 1;
    // let cell = svg.select(`.row.row-${layer}`).select(`.cell.col-${head}`);
    let decoderCanvas, crossCanvas;


    if (attnPattern.select('canvas').size() === 0) {
      let foreignObject = attnPattern.append('foreignObject')
        .attr('width', width)
        .attr('height', height)
        .attr('class', 'attn-pattern')
        .attr('visibility', 'visible');

      let foBody = foreignObject.append('xhtml:body')
        .attr('width', width)
        .attr('height', height)
        .style('background-color', 'none');

      crossCanvas = foBody.append('canvas')
        .attr('width', width)
        .attr('height', height / 2)
        .style('border', '0.1px solid black')
        .attr('id', 'cross-attn-canvas');
      decoderCanvas = foBody.append('canvas')
        .attr('width', width)
        .attr('height', height / 2)
        .style('border', '0.1px solid black')
        .attr('id', 'decoder-attn-canvas');
    } else {
      decoderCanvas = attnPattern.select('#decoder-attn-canvas');
      crossCanvas = attnPattern.select('#cross-attn-canvas');
    }



    // console.log();
    // if (cell.select('foreignObject').length())

    
        // .append("xhtml:body");


    // let layer = Math.floor(idx / 12);
    // let head = idx % 12;

    // let context = canvas.node().getContext('2d');
    let context, imageData, image, bitmapOptions, resizedImage;

    // Decoder attention heatmap
    context = decoderCanvas.node().getContext('2d');
    imageData = new Uint8ClampedArray;
    imageData = Uint8ClampedArray.from(decoder_attn);
    image = context.createImageData(outputLen, outputLen);
    image.data.set(imageData);
    bitmapOptions = {
      'resizeWidth': Math.round(decoderAttnWidth),
      'resizeHeight': Math.round(decoderAttnHeight),
    }
    resizedImage = await window.createImageBitmap(image, 0, 0, image.width, image.height, bitmapOptions);
    context.clearRect(0, 0, width, height / 2);
    context.drawImage(resizedImage, decoderAttnMargin, 0);

    // Cross attention heatmap
    context = crossCanvas.node().getContext('2d');
    imageData = new Uint8ClampedArray;
    imageData = Uint8ClampedArray.from(cross_attn);
    image = context.createImageData(outputLen, inputLen);
    image.data.set(imageData);
    bitmapOptions = {
      'resizeWidth': Math.round(crossAttnWidth),
      'resizeHeight': Math.round(crossAttnHeight),
    }
    resizedImage = await window.createImageBitmap(image, 0, 0, image.width, image.height, bitmapOptions);
    context.clearRect(0, 0, width, height / 2);
    context.drawImage(resizedImage, crossAttnMargin, 0);



    attnPattern.append('text')
      .attr('class', 'fas prune-button fa-4x')
      .text('\uf057')
      .attr('x', marginX / 2 + innerWidth)
      .attr('y',  3 * marginY)
      .raise()
      .on('click', function(){
        attnPattern.selectAll('*').remove();
        d3.select(`#${headType}-heatmap`).style("visibility", 'visible');
      });

    return state;
  }

    // canvas.node().addEventListener('click', function(event){
    //   console.log(event)
    // }, false);

  // // let canvas = d3.select('#attentionView').append('canvas')
  // //   .attr("id", "attention-canvas")
  // //   .attr('width', innerWidth)
  // //   .attr('height', innerHeight)
  // //   .style("width", innerWidth + "px")
  // //   .style("height", innerHeight + "px")
  // //   .style('margin-left', marginX + 'px')
  // //   .style('margin-top', marginY + 'px');

  // attn_patten.forEach((attn, i) => {
  //   drawImage(attn);
  // })

  // cellsEnter.append('rect')
  //   .attr('class', 'border')
  //   .attr('height', yScale.bandwidth())
  //   .attr('width', xScale.bandwidth())
  //   .attr('ry', yScale.bandwidth() * 0.1)
  //   .attr('rx', xScale.bandwidth() * 0.1)
  //   .raise();

  // cellsEnter.append('text')
  //   .attr('class', 'fas prune-button')
  //   .text('\uf057')
  //   .attr('x', (4 / 5) * xScale.bandwidth())
  //   .raise()
  //   .on('click', function(){
  //     let layerIdx = d3.select(this.parentNode.parentNode).attr('class').match(/(\d+)/)[0] - 1;
  //     let headIdx = d3.select(this.parentNode).attr('class').match(/(\d+)/)[0] - 1;
  //     // state.pruned_heads

  //     d3.select(this.parentNode).remove()
  //     if (layerIdx in state.pruned_heads){
  //       state.pruned_heads[layerIdx].push(headIdx);
  //     } else {
  //       state.pruned_heads[layerIdx] = [headIdx];
  //     }
  //     console.log(state)
  //   });

  //   // let layerIdx, headIdx;
  //   // console.log(parseInt(Object.keys(state.pruned_heads)[0]))
  // Object.keys(state.pruned_heads).forEach(layerIdx => {
  //   state.pruned_heads[layerIdx].forEach(headIdx => {
  //     d3.select(`.row-${parseInt(layerIdx) + 1} > .col-${parseInt(headIdx) + 1}`).remove()
  //   })
  // })

  return state;
}