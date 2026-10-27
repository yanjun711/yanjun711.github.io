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
    description: 'We use language supervision during training to improve humanoid loco-manipulation, without changing the model at inference.',
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
    description: 'We check whether a previously validated motion can be reused from the current robot state, and learn which motions to keep in memory.',
    project: 'https://aigeeksgroup.github.io/HumanoidTTT',
    code: 'https://github.com/AIGeeksGroup/HumanoidTTT',
    image: '/images/humanoidttt.webp',
    preview: { src: '/videos/publications/humanoidttt.mp4', poster: '/videos/publications/humanoidttt-poster.webp', alt: 'Real-world humanoid waving and squatting, excerpted from the HumanoidTTT project demonstrations.' },
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
    description: 'We separate pose and velocity information into two token streams to generate humanoid motion from scene images and text.',
    project: 'https://aigeeksgroup.github.io/MotionVLA',
    code: 'https://github.com/AIGeeksGroup/MotionVLA',
    image: '/images/motionvla.webp',
    preview: { src: '/videos/publications/motionvla.mp4', poster: '/videos/publications/motionvla-poster.webp', alt: 'Real-world humanoid turning and walking, excerpted from the MotionVLA project demonstrations.' },
    imageAlt: 'MotionVLA paper figure comparing generated humanoid motion sequences.',
  },
];

export type Experience = {
  organization: string;
  role: string;
  date: string;
  url: string;
  logo: string;
  logoAlt: string;
  advisor?: string;
  advisorUrl?: string;
  topic?: string;
};

export const researchExperience: Experience[] = [
  { organization: 'EGI Lab, Peking University', role: 'Research Assistant', date: 'Jan 2026 — Present', advisor: 'Prof. Hao Tang', advisorUrl: 'https://ha0tang.github.io/', topic: 'World models', url: 'https://www.pku.edu.cn/', logo: '/images/logos/pku.webp', logoAlt: 'Peking University emblem' },
  { organization: 'Vision Lab, HUST', role: 'Research Intern', date: 'Jan — May 2025', advisor: 'Prof. Xinggang Wang', advisorUrl: 'https://xwcv.github.io/', topic: 'Efficient VLA', url: 'https://www.hust.edu.cn/', logo: '/images/logos/hust.webp', logoAlt: 'Huazhong University of Science and Technology emblem' },
];

export const internshipExperience: Experience[] = [
  { organization: 'AI² Robotics', role: 'Research Intern', date: 'May 2026 — Present', url: 'https://ai2robotics.com/', logo: '/images/logos/ai2.webp', logoAlt: 'AI² Robotics logo' },
  { organization: 'Beijing Academy of Artificial Intelligence', role: 'Research Intern', date: 'Jan — May 2026', url: 'https://www.baai.ac.cn/', logo: '/images/logos/baai.webp', logoAlt: 'Beijing Academy of Artificial Intelligence logo' },
];

export function bibtex(paper: typeof publications[number]) {
  return `@misc{${paper.id.replaceAll('-', '')}${paper.year},\n  title = {${paper.title}},\n  author = {${paper.authors.join(' and ')}},\n  year = {${paper.year}},\n  eprint = {${paper.arxiv}},\n  archivePrefix = {arXiv},\n  url = {https://arxiv.org/abs/${paper.arxiv}}\n}`;
}
