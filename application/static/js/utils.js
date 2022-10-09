import { renderInstanceView } from './instance-view.js';
import { renderImportanceFromState } from './component-view.js';
// import { renderProjection } from './projection-view.js';



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
