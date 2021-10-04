export const renderImportanceFromState = (attentionSVG, state) => {
  let value = state.attentionScale;
  if (value === 'aggregate') {
    state = renderImportance(state.aggregate_importance, state.aggregate_pattern, attentionSVG, 700, 500, state);
  } else {
    state = renderImportance(state.instance_importance, state.instance_pattern, attentionSVG, 700, 500, state);
  }
  return state;
}


export const renderImportance = (data, attn_patten, svg, width, height, state) => {

  // d3.select('#attentionView').style('position', 'relative');
  // svg.style("position", "absolute")
  let attn_len = (attn_patten[0].attn.length / 4)**(1/2)
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
    .range([marginY, marginY + innerHeight])
    .padding(0.05);

  // Axis
  svg.selectAll('g').remove()

  svg.append('g')
    .attr('class', 'x-axis')
    .attr('transform', `translate(0, ${yScale.range()[1]})`)
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
  d3.selectAll('.x-axis > .tick')
    .attr('class', d => `tick head-${d}`);
  d3.selectAll('.y-axis > .tick')
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

      // Check if the head has been pruned
      if (d3.select(this.parentNode).size() === 0) {
        return
      }

      let head = d3.select(this).attr('class').match(/(\d+)/)[0];
      let layer = d3.select(this.parentNode).attr('class').match(/(\d+)/)[0];
      state.attentionIdx = [layer, head];

      // Unselect previous selections 
      d3.selectAll('.cell > .selected').classed('selected', false);
      d3.selectAll('.x-axis > .tick.selected').classed('selected', false);
      d3.selectAll('.y-axis > .tick.selected').classed('selected', false);

      // Select heatmap cell and tick on axis
      d3.select(this).classed('selected', true);
      d3.select(this).select('.border').classed('selected', true);
      d3.select(this).select('.prune-button').classed('selected', true);
      d3.select(`.x-axis > .tick.head-${head}`).classed('selected', true);
      d3.select(`.y-axis > .tick.layer-${layer}`).classed('selected', true);
      if (state.tokenIdx != null) {
        renderColor(state.attention[layer - 1][head - 1][state.tokenIdx]);
      }
    }).on('mouseover', function() {
      d3.select(this).select('.prune-button').classed('hovered', true);
      d3.select(this).select('.border').classed('hovered', true);
    }).on('mouseleave', function() {
      d3.select(this).select('.prune-button').classed('hovered', false);
      d3.select(this).select('.border').classed('hovered', false);
    })

  let rect = cellsEnter.append('rect').merge(cells.selectAll('rect'));
  let rectEnter = rect.enter();


  rectEnter.merge(rect)
    // .attr('class', (d, i) => `head-${i + 1}`)
    .attr('height', yScale.bandwidth())
    .attr('width', xScale.bandwidth())
    .attr('ry', yScale.bandwidth() * 0.1)
    .attr('rx', xScale.bandwidth() * 0.1)
    .attr('fill', d => d3.interpolateReds(d))
    // .on('click', clickHead)

  cellsEnter.append('text').merge(cells.selectAll('text'))
    .attr('dx', `${xScale.bandwidth() / 4}`)
    .attr('dy', `${yScale.bandwidth() / 2}`)
    .style('font-size', `${yScale.bandwidth() / 3.5}px`)
    .text(d => d.toFixed(2));

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
    let layer = attn.layer + 1;
    let head = attn.head + 1;
    let cell = d3.select(`.row.row-${layer}`).select(`.cell.col-${head}`);

    let canvas;
    let width = xScale.bandwidth();
    let height = yScale.bandwidth()

    if (cell.select('canvas').size() === 0) {
      let foreignObject = cell.append('foreignObject')
        .attr('width', width)
        .attr('height', height)
        .attr('class', 'attn-pattern')
        .attr('visibility', (state.attentionView === 'pattern')? 'visible' : 'hidden');

      let foBody = foreignObject.append('xhtml:body')
        .attr('width', width)
        .attr('height', height)
        .style('background-color', 'none');

      canvas = foBody.append('canvas')
        .attr('width', width)
        .attr('height', height)
        .style('border', '0.1px solid black');
    } else {
      canvas = cell.select('canvas');
    }
    // console.log();
    // if (cell.select('foreignObject').length())

    
        // .append("xhtml:body");

    let context = canvas.node().getContext('2d');

    // let layer = Math.floor(idx / 12);
    // let head = idx % 12;

    // let context = canvas.node().getContext('2d');

    let imageData = new Uint8ClampedArray;
    imageData = Uint8ClampedArray.from(attn.attn);
    let image = context.createImageData(attn_len, attn_len);
    image.data.set(imageData);
    let bitmapOptions = {
      'resizeWidth': Math.round(xScale.bandwidth()),
      'resizeHeight': Math.round(yScale.bandwidth()),
    }

    let resizedImage = await window.createImageBitmap(image, 0, 0, image.width, image.height, bitmapOptions);

    context.clearRect(0, 0, height, height);
    context.drawImage(resizedImage, 0, 0);

    // canvas.node().addEventListener('click', function(event){
    //   console.log(event)
    // }, false);
  }

  // let canvas = d3.select('#attentionView').append('canvas')
  //   .attr("id", "attention-canvas")
  //   .attr('width', innerWidth)
  //   .attr('height', innerHeight)
  //   .style("width", innerWidth + "px")
  //   .style("height", innerHeight + "px")
  //   .style('margin-left', marginX + 'px')
  //   .style('margin-top', marginY + 'px');

  attn_patten.forEach((attn, i) => {
    drawImage(attn);
  })

  cellsEnter.append('rect')
    .attr('class', 'border')
    .attr('height', yScale.bandwidth())
    .attr('width', xScale.bandwidth())
    .attr('ry', yScale.bandwidth() * 0.1)
    .attr('rx', xScale.bandwidth() * 0.1)
    .raise();

  cellsEnter.append('text')
    .attr('class', 'fas prune-button')
    .text('\uf057')
    .attr('x', (4 / 5) * xScale.bandwidth())
    .raise()
    .on('click', function(){
      let layerIdx = d3.select(this.parentNode.parentNode).attr('class').match(/(\d+)/)[0] - 1;
      let headIdx = d3.select(this.parentNode).attr('class').match(/(\d+)/)[0] - 1;
      // state.pruned_heads

      d3.select(this.parentNode).remove()
      if (layerIdx in state.pruned_heads){
        state.pruned_heads[layerIdx].push(headIdx);
      } else {
        state.pruned_heads[layerIdx] = [headIdx];
      }
      console.log(state)
    });

    // let layerIdx, headIdx;
    // console.log(parseInt(Object.keys(state.pruned_heads)[0]))
    Object.keys(state.pruned_heads).forEach(layerIdx => {
      state.pruned_heads[layerIdx].forEach(headIdx => {
        d3.select(`.row-${parseInt(layerIdx) + 1} > .col-${parseInt(headIdx) + 1}`).remove()
      })
    })
  return state;
}