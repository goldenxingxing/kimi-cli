import type { ManagedSkill } from "../../lib/api/apis/AdminSkillsApi";

/**
 * A skill's category is declared when it can be, derived when it cannot.
 *
 * A SKILL.md may name its own `category` in frontmatter (top level or under
 * `metadata`), and that always wins — it is the only way an uploaded skill can
 * state where it belongs. Most skills declare nothing, so the fallback
 * classifies from the two things every skill does have: name and description.
 */
export type KnownCategoryId =
  | "agent"
  | "legal"
  | "finance"
  | "devops"
  | "product"
  | "marketing"
  | "data"
  | "research"
  | "engineering"
  | "design"
  | "learning"
  | "comms"
  | "docs"
  | "writing"
  | "other";

/** A known id, or `custom:<slug>` for a category a skill declared itself. */
export type SkillCategoryId = KnownCategoryId | (string & {});

export interface SkillCategory {
  id: KnownCategoryId;
  label: string;
}

/** Display order of the category sections. `other` always sorts last. */
export const SKILL_CATEGORIES: SkillCategory[] = [
  { id: "engineering", label: "Engineering" },
  { id: "devops", label: "DevOps & Infra" },
  { id: "data", label: "Data & Analytics" },
  { id: "design", label: "Design & Media" },
  { id: "product", label: "Product & Strategy" },
  { id: "marketing", label: "Marketing & Growth" },
  { id: "finance", label: "Finance & Investing" },
  { id: "research", label: "Research & Academia" },
  { id: "writing", label: "Writing & Content" },
  { id: "comms", label: "Communication" },
  { id: "docs", label: "Docs & Office Files" },
  { id: "legal", label: "Legal & Compliance" },
  { id: "learning", label: "Learning & Career" },
  { id: "agent", label: "Agent Tooling" },
  { id: "other", label: "Other" },
];

const CATEGORY_RANK = new Map<string, number>(
  SKILL_CATEGORIES.map((category, index) => [category.id, index]),
);

export const SKILL_CATEGORY_LABELS = new Map<string, string>(
  SKILL_CATEGORIES.map((category) => [category.id, category.label]),
);

const CUSTOM_PREFIX = "custom:";

/**
 * What a declared `category:` may say and still land in a built-in section.
 *
 * Skills are written by many hands, so the spelling on the label is not the
 * only reasonable thing to type. Anything not listed here — and not an id or a
 * label — still counts: it becomes a section of its own rather than being
 * silently dropped into the wrong one.
 */
const CATEGORY_ALIASES: Record<string, KnownCategoryId> = {
  dev: "engineering",
  development: "engineering",
  coding: "engineering",
  software: "engineering",
  programming: "engineering",
  ops: "devops",
  sre: "devops",
  infra: "devops",
  infrastructure: "devops",
  platform: "devops",
  analytics: "data",
  "data-science": "data",
  database: "data",
  bi: "data",
  media: "design",
  video: "design",
  creative: "design",
  ux: "design",
  pm: "product",
  strategy: "product",
  planning: "product",
  growth: "marketing",
  sales: "marketing",
  seo: "marketing",
  advertising: "marketing",
  investing: "finance",
  investment: "finance",
  trading: "finance",
  academia: "research",
  science: "research",
  analysis: "research",
  content: "writing",
  copywriting: "writing",
  translation: "writing",
  communication: "writing",
  comms: "comms",
  email: "comms",
  productivity: "comms",
  documents: "docs",
  office: "docs",
  documentation: "docs",
  compliance: "legal",
  security: "legal",
  education: "learning",
  training: "learning",
  career: "learning",
  meta: "agent",
  tooling: "agent",
  misc: "other",
  uncategorized: "other",
};

/** Lowercase, collapse separators, drop "&" — "Design & Media" -> "design-media". */
function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/&/g, " ")
    .replace(/[\s_/]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

const LABEL_SLUGS = new Map<string, KnownCategoryId>(
  SKILL_CATEGORIES.map((category) => [slugify(category.label), category.id]),
);

/**
 * Resolve a declared `category:` string to a section id.
 *
 * An unrecognized value is kept rather than discarded: an install may run its
 * own taxonomy, and losing the declaration would be worse than showing one
 * extra section.
 */
export function resolveDeclaredCategory(declared: string): SkillCategoryId | null {
  const slug = slugify(declared);
  if (!slug) return null;
  if (CATEGORY_RANK.has(slug)) return slug as KnownCategoryId;
  return LABEL_SLUGS.get(slug) ?? CATEGORY_ALIASES[slug] ?? `${CUSTOM_PREFIX}${slug}`;
}

/**
 * Ordered classification rules — **first match wins**, so the order encodes the
 * precedence between overlapping vocabularies. `equity-research-report` is
 * finance rather than research because finance is tested first; `seo-content-
 * writer` is marketing rather than writing for the same reason.
 */
const RULES: { id: SkillCategoryId; pattern: RegExp }[] = [
  // Skills about the agent itself, before their topical words can claim them.
  {
    id: "agent",
    pattern:
      /^kimi-|\bskill-creator\b|\bskills?-finder\b|\bfind-skills\b|\bsubagent\b|\bsuperpowers\b|\bdispatching-parallel-agents\b|\b(writing|executing)-plans\b|\bverification-before-completion\b|\bbrainstorming\b|\bmedia-use\b/,
  },
  {
    // `tos` is anchored: a bare \btos\b also matches "how-tos".
    id: "legal",
    pattern:
      /\blegal\b|^tos\b|\bterms[ -]of[ -]service\b|\bcompliance\b|\bregulatory\b|\bnmpa\b|\biso-27001\b|\bcontract\b|\bgdpr\b|\blitigation\b/,
  },
  {
    id: "finance",
    pattern:
      /\bfinanc(e|ial)\b|\bequity\b|\bstock\b|\betf\b|\bfund(s|raising)?\b|\binvest(ing|ment|or)?\b|\bvaluation\b|\bcashflow\b|\bdcf\b|\bearnings\b|\bcommodit(y|ies)\b|\btrading\b|\bbacktest\b|\bportfolio\b|\bhedge\b|\bcn-finance\b/,
  },
  {
    id: "devops",
    pattern:
      /\bk8s\b|\bkube(ctl|rnetes)?\b|\bterraform\b|\bdocker\b|\bcluster\b|\bincident\b|\bdeploy(ment)?\b|\bsre\b|\binfra(structure)?\b|\bobservability\b|\blog-(diagnostic|error)\b|\bload-(test|profil)/,
  },
  {
    id: "product",
    pattern:
      /\bsaas\b|\bokr\b|\bprd\b|\bproduct-spec\b|\bsprint\b|\biteration\b|\broadmap\b|\bbacklog\b|\bgantt\b|\btimeline\b|\bmilestone\b|\bpricing\b|\bpitch-deck\b|\bbusiness-plan\b|\bproject-sizing\b|\bworkload\b|\buser-story\b|\bstory-map\b|\bidea-to-prd\b/,
  },
  {
    id: "learning",
    pattern:
      /\btutor\b|\bmentor\b|\bquiz\b|\bflashcards?\b|\banki\b|\binterview\b|\bexam\b|\blessons?\b|\bcurriculum\b|\bbloom\b|\bteach(ing)?\b|\bresume\b|\bcv-tailor\b|\bcareer\b|\bonboarding\b/,
  },
  {
    id: "marketing",
    pattern:
      /\bseo\b|\bads\b|\bad-(copy|creative|campaign)|\badvertis|\bcampaign\b|\becom(merce)?\b|\bmarketing\b|\bbrand\b|\bcopywrit|\bnewsletter\b|\bgrowth\b|\bchurn\b|\bretention\b|\blisting\b|\bviral\b|\bthread\b|\bxhs\b|\bwechat\b|\bzhihu\b|\blanding-page\b|\blp-proto\b|\bconversion\b/,
  },
  {
    id: "data",
    pattern:
      /\bdata\b|\bdataset\b|\bdatabase\b|\bsql\b|\bchart\b|\bviz\b|\bvisuali[sz]|\bstat(s|istic|istics|istical)?\b|\bhypothes[ie]s\b|\bregression\b|\bcorrelation\b|\bcorr\b|\boutlier\b|\bmetrics?\b|\bscorer\b|\bscoring\b|\bscorecard\b|\bsplit-test\b|\ba\/b\b|\bheatmap\b|\bquery\b/,
  },
  {
    id: "research",
    pattern:
      /\bresearch\b|\bmarket-insight\b|\bpaper\b|\bscholar|\bscientific\b|\bsci-paper\b|\bcitations?\b|\bcite\b|\bref-style\b|\bacademic\b|\bthesis\b|\bsurvey\b|\bcompetit(or|ive|ors)\b|\bastro\b|\bobservation\b|\bsun(light|-path)\b|\bcross-examine\b|\bsearch-expert\b|\bdeep-probe\b/,
  },
  {
    id: "engineering",
    pattern:
      /\bcode\b|\bcoding\b|\bvibecoding\b|\bdebug|\brefactor|\btests?\b|\btesting\b|\btdd\b|\bcommit\b|\bgit(lab|hub)?\b|\bworktrees?\b|\brepo(sitory)?\b|\bbranch\b|\bapi\b|\bopenapi\b|\bbackend\b|\bfrontend\b|\bwebapp\b|\bserver\b|\bscraper\b|\bscrap(e|ing)\b|\bbrowser\b|\bplaywright\b|\bperf(ormance)?\b|\bvuln(erabilit(y|ies))?\b|\bsecurity\b|\bsecure\b|\bdev-guide\b|\bddd\b|\bglossary\b|\bhttp\b|\bdownload\b|\bupload\b|\bbrowse\b|\blocal(e|ization)\b|\bi18n\b|\bwidget\b/,
  },
  {
    // The CJK tokens matter here: a good share of the media skills describe
    // themselves only in Chinese, and would otherwise land in "Other".
    id: "design",
    pattern:
      /\bvideo\b|\bhyperframes?\b|\bremotion\b|\bmotion\b|\banimation\b|\bcaptions?\b|\bsubtitles?\b|\brecut\b|\bexplainer\b|\btalking-head\b|\blower-third|\boverlays?\b|\bfootage\b|视频|图片|抠图|插画|海报|时间线|幻灯片|配色|\bslides?\b|\bslideshow\b|\bkeynote\b|\bdeck\b|\bimage\b|\bphoto\b|\bsketch\b|\billustration\b|\bmagazine\b|\binfographic\b|\bdiagram\b|\bui\b|\bux\b|\btheme\b|\bdesign\b|\btts\b|\bspeech-synthesis\b|\bvoice(over)?\b|\bpodcast\b|\bmusic\b|\baudio\b|\bportrait\b|\bfashion\b|\bposter\b|\bmockup\b/,
  },
  {
    id: "comms",
    pattern:
      /\bemails?\b|\bmail(er)?\b|\bimap\b|\bsmtp\b|\bwhatsapp\b|\bslack\b|\bcalendar\b|\bmeetings?\b|\bminutes\b|\brecap\b|\bstandup\b|\bdaily-report|\bsupport-response\b|\bcustomer-reply\b|\badhd\b|\baudience\b|\bspeech-craft\b|\brhetoric\b/,
  },
  {
    id: "docs",
    pattern:
      /\bpdf\b|\bdocx?\b|\bxlsx\b|\bpptx\b|\bspreadsheet\b|\bsop\b|\bprocess-doc\b|\bdocument(ation|s)?\b|\bmanual\b|\btemplate\b|\bmarkdown\b/,
  },
  {
    id: "writing",
    pattern:
      /\bwrit(e|er|ers|ing)\b|\bedit(or|ing)\b|\btranslat|\bhumaniz|\blongread\b|\bnarrative\b|\bstory(telling)?\b|\bblog\b|\bposts?\b|\bscripts?\b|\bsummar(y|ize|ise)\b|\breports?\b|\bprose\b|\bxindaya\b/,
  },
];

/**
 * Classify one skill.
 *
 * A declared `category:` is authoritative — that is how an uploaded skill says
 * where it belongs. Otherwise the name is matched on its own first (the
 * strongest signal), and only then the name plus description together, so a
 * passing mention of "code" in prose cannot outrank the skill's own subject.
 */
export function categorizeSkill(
  skill: Pick<ManagedSkill, "name" | "description" | "category">,
): SkillCategoryId {
  if (skill.category) {
    const declared = resolveDeclaredCategory(skill.category);
    if (declared) return declared;
  }
  const name = skill.name.toLowerCase();
  for (const rule of RULES) {
    if (rule.pattern.test(name)) return rule.id;
  }
  const text = `${name} ${skill.description ?? ""}`.toLowerCase();
  for (const rule of RULES) {
    if (rule.pattern.test(text)) return rule.id;
  }
  return "other";
}

/**
 * Position of a category in the section order.
 *
 * Self-declared categories sit just above "Other": they are more specific than
 * the catch-all, but they are not part of the curated order, so they should not
 * push a built-in section around. The fraction keeps that without renumbering.
 */
export function categoryRank(id: SkillCategoryId): number {
  const known = CATEGORY_RANK.get(id);
  if (known !== undefined) return known;
  return (CATEGORY_RANK.get("other") ?? SKILL_CATEGORIES.length) - 0.5;
}

export function categoryLabel(id: SkillCategoryId): string {
  const known = SKILL_CATEGORY_LABELS.get(id);
  if (known) return known;
  if (!id.startsWith(CUSTOM_PREFIX)) return "Other";
  return id
    .slice(CUSTOM_PREFIX.length)
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export interface SkillGroup {
  id: SkillCategoryId;
  label: string;
  skills: ManagedSkill[];
}

/**
 * Collapse an already category-sorted list into consecutive runs.
 *
 * Ordering stays the sole responsibility of `sortSkills`; this only slices the
 * result, so a list sorted any other way yields a single group per run — which
 * is exactly what the non-grouped views want.
 */
export function groupByCategory(skills: ManagedSkill[]): SkillGroup[] {
  const groups: SkillGroup[] = [];
  for (const skill of skills) {
    const id = categorizeSkill(skill);
    const last = groups[groups.length - 1];
    if (last && last.id === id) last.skills.push(skill);
    else groups.push({ id, label: categoryLabel(id), skills: [skill] });
  }
  return groups;
}
