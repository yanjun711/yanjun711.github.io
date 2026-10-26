import { publications, bibtex } from '../../data/site';
export function getStaticPaths() {
  return publications.map(paper => ({ params: { id: paper.id }, props: { paper } }));
}
export function GET({ props }: { props: { paper: typeof publications[number] } }) {
  return new Response(bibtex(props.paper), { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
}
