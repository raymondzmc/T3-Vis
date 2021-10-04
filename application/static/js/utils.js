import { renderInstanceView } from './instance-view.js';
import { renderImportanceFromState } from './component-view.js';

export const selectExample = (id, state) => {
  $('#loader').show();
  let exampleID = (typeof(id) === 'string')? +id.split('-')[1]: id;

  // // Unselect all previous selections
  // d3.selectAll('.example.selected')
  //   .classed('selected', false)
  //   .attr('r', 3);

  // // Select current example by ID
  // d3.select(`.example#example-${exampleID}`)
  //   .classed('selected', true)
  //   .attr('r', 10);
  state.selectedIdx.clear();
  state.selectedIdx.add(id);

  state['example_id'] = exampleID;

  // if (state['checkpoint'] === 0 ){
  //   $('#loader').hide();
  //   return
  // }
  
  const server_query = d3.json('../api/eval_one', {
      method: "POST",
      body: JSON.stringify(state),
      headers: {
          "Content-type": "application/json; charset=UTF-8"
      }
  })

  server_query.then(response => {
      state['attention'] = response['attn'];
      let importance = response['head_importance'];
      let attn_patten = response['attn_pattern'];

      state.instance_importance = importance;
      state.instance_pattern = attn_patten;
      $("#instanceAttention").prop("disabled", false);
      state.tokenIdx = null;
      state = renderInstanceView(
        response['tokens'],
        response['output'],
        response['input_saliency'],
        response['label'],
        response['loss'],
        '#input-token-container',
        '#output-token-container',
        state,
      );

      let attentionSVG = d3.select("#attention-svg");

      state = renderImportanceFromState(attentionSVG, state);
      $('#loader').hide();
  });
  return state;
}

export const scrollContent = id => {
  // console.log(id, typeof(id));
  let exampleID = (typeof(id) === 'string')? +id.split('-')[1]: id;

  let contentView = document.getElementById('contentView');
  let contentElement = document.getElementById(`content-${exampleID}`);

  // // Define offsets from the top and scroll to the respective content
  let targetOffset = contentElement.offsetTop - contentView.offsetTop + 50;
  let offset = contentView.scrollTop;

  let contentContainer = d3.select('#contentView');
  contentContainer
    .transition()
    .duration(500)
    .tween('scroll', () => {
      let i = d3.interpolateNumber(offset, targetOffset);
      return (t) => {
        contentView.scrollTop = i(t);
      };
    });

  highlightContent(id);
};

export const highlightContent = id => {
  let exampleID = (typeof(id) === 'string')? +id.split('-')[1]: id;

  // Highlight respective content in the conversation list
  let contentContainer = d3.select('#contentView');
  contentContainer
    .selectAll('.content.selected')
    .classed('selected', false);
  contentContainer
    .select(`#content-${exampleID}`)
    .classed('selected', true);
}

const appendOnce = (selection, element) => {
  let l = element.split('.');
  let g = selection.selectAll(element).data([0]);

  g.enter().append(l[0]).attr('class', l.slice(1).join(' '))

  return g;
}


export const checkArrays = (a, b) => {
  let setA = new Set(a);
  let setB = new Set(b);
  if (setA.size !== setB.size) return false;
  for (const a of setA) if (!setB.has(a)) return false;
  return true;
}
