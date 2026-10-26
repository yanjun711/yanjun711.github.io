export const profile = {
  name: 'Yanjun Li',
  email: 'liyanjun@hust.edu.cn',
  github: 'https://github.com/yanjun711',
  scholar: 'https://scholar.google.com/citations?user=xpyQJSAAAAAJ&hl=en',
  description: 'Yanjun Li is an undergraduate student at Huazhong University of Science and Technology studying robot learning and world models.',
};

export const publications = [
  {
    id: 'language-forcing',
    name: 'Language-Forcing',
    title: 'Language-Forcing: Advancing Humanoid Loco-Manipulation via Semantic Alignment of Whole-Body World Action Models',
    authors: ['Yanjun Li', 'Jiarui Ye', 'Weiyu Guo', 'Zixiao He', 'Kangjie Ye', 'Dihan Wang', 'Yishuo Cheng', 'Zishang Xiang', 'Mingju Gao', 'Hao Tang'],
    year: 2026,
    arxiv: '2610.23894',
    topic: 'World action models',
    equal: false,
    description: 'Language supervision helps whole-body world action models focus on task-relevant changes, with no additional inference overhead.',
    project: '',
    code: '',
    image: '/images/language-forcing.webp',
    imageAlt: 'Language-Forcing paper figure showing humanoid tasks and results.',
  },
  {
    id: 'humanoidttt',
    name: 'HumanoidTTT',
    title: 'HumanoidTTT: Test-Time Capability Reuse for Efficient Humanoid Control',
    authors: ['Jingtai Yang', 'Yining Wu', 'Yanjun Li', 'Zeyu Zhang', 'Hao Tang'],
    year: 2026,
    arxiv: '2610.00198',
    topic: 'Test-time learning',
    equal: false,
    description: 'Reuse validated humanoid motions from applicable entry states, and adapt a bounded capability memory through deployment feedback.',
    project: 'https://aigeeksgroup.github.io/HumanoidTTT',
    code: 'https://github.com/AIGeeksGroup/HumanoidTTT',
    image: '/images/humanoidttt.webp',
    imageAlt: 'HumanoidTTT paper diagram showing motion reuse and capability consolidation.',
  },
  {
    id: 'motionvla',
    name: 'MotionVLA',
    title: 'MotionVLA: Vision-Language-Action Model for Humanoid Motion',
    authors: ['Nonghai Zhang', 'Siyu Zhai', 'Yanjun Li', 'Zeyu Zhang', 'Zhihan Yin', 'Yandong Guo', 'Boxin Shi', 'Hao Tang'],
    year: 2026,
    arxiv: '2606.15142',
    topic: 'Humanoid motion',
    equal: true,
    description: 'A vision-language-action model with dual-stream frequency tokenization for humanoid motion conditioned on scenes and language.',
    project: 'https://aigeeksgroup.github.io/MotionVLA',
    code: 'https://github.com/AIGeeksGroup/MotionVLA',
    image: '/images/motionvla.webp',
    imageAlt: 'MotionVLA paper figure comparing generated humanoid motion sequences.',
  },
];

export const experience = [
  { organization: 'AI² Robotics', role: 'Research Intern', date: 'May 2026 — Present', description: 'World models for humanoid policies.' },
  { organization: 'EGI Lab, Peking University', role: 'Research Assistant', date: 'Jan 2026 — Present', description: 'Robot learning and world action models. Advised by Prof. Hao Tang.' },
  { organization: 'Beijing Academy of Artificial Intelligence', role: 'Research Intern', date: 'Jan — May 2026', description: 'Memory for embodied agents and dexterous skill learning.' },
  { organization: 'Vision Lab, HUST', role: 'Research Intern', date: 'Jan — May 2025', description: 'Efficient vision-language-action policies. Advised by Prof. Xinggang Wang.' },
];

export function bibtex(paper: typeof publications[number]) {
  return `@misc{${paper.id.replaceAll('-', '')}${paper.year},\n  title = {${paper.title}},\n  author = {${paper.authors.join(' and ')}},\n  year = {${paper.year}},\n  eprint = {${paper.arxiv}},\n  archivePrefix = {arXiv},\n  url = {https://arxiv.org/abs/${paper.arxiv}}\n}`;
}
