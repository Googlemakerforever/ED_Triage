import { PrismaClient } from "@prisma/client";
import bcrypt from "bcryptjs";

const prisma = new PrismaClient();

type Choice = { label: string; text: string; isCorrect: boolean };

async function upsertQuestion(topicId: string, stem: string, difficulty: number, explanation: string, wrongRationale: string, mistakeRule: string, choices: Choice[]) {
  const existing = await prisma.question.findFirst({ where: { topicId, stem } });
  if (existing) return existing;

  return prisma.question.create({
    data: {
      topicId,
      stem,
      difficulty,
      explanation,
      wrongRationale,
      mistakeRule,
      answerChoices: { create: choices }
    }
  });
}

async function main() {
  const passwordHash = await bcrypt.hash("Str0ngPassw0rd!", 12);

  const user = await prisma.user.upsert({
    where: { email: "demo@aegismcat.com" },
    update: { passwordHash, fullName: "Demo Student" },
    create: {
      email: "demo@aegismcat.com",
      fullName: "Demo Student",
      passwordHash,
      profile: {
        create: {
          targetScore: 525,
          diagnosticScore: 508,
          strongestSection: "BB",
          weakestSection: "CARS",
          weeklyStudyHours: 24,
          studyPhase: "intensive",
          testDate: new Date("2026-08-15")
        }
      }
    }
  });

  if (!(await prisma.userProfile.findUnique({ where: { userId: user.id } }))) {
    await prisma.userProfile.create({ data: { userId: user.id, targetScore: 525, weeklyStudyHours: 24, studyPhase: "intensive" } });
  }

  const topics = await Promise.all([
    prisma.topic.upsert({
      where: { slug: "enzyme-kinetics" },
      update: {
        blueprintCategory: "Biological and Biochemical Foundations",
        reasoningSkill: "Scientific Reasoning and Problem Solving",
        timingBurden: 3
      },
      create: {
        slug: "enzyme-kinetics",
        title: "Enzyme Kinetics",
        section: "BB",
        summary: "Michaelis-Menten modeling, inhibition patterns, and data interpretation.",
        highYield: true,
        contentMd: "Focus on Km/Vmax shifts across inhibitor classes and graph transformations.",
        blueprintCategory: "Biological and Biochemical Foundations",
        reasoningSkill: "Scientific Reasoning and Problem Solving",
        timingBurden: 3
      }
    }),
    prisma.topic.upsert({
      where: { slug: "cars-argument-structure" },
      update: {
        blueprintCategory: "CARS",
        reasoningSkill: "Reasoning Within the Text",
        timingBurden: 4
      },
      create: {
        slug: "cars-argument-structure",
        title: "CARS Argument Structure",
        section: "CARS",
        summary: "Map claims, evidence, assumptions, and author stance.",
        highYield: true,
        contentMd: "Track paragraph function and avoid outside knowledge during elimination.",
        blueprintCategory: "CARS",
        reasoningSkill: "Reasoning Within the Text",
        timingBurden: 4
      }
    }),
    prisma.topic.upsert({
      where: { slug: "electrochemistry" },
      update: {
        blueprintCategory: "Chemical and Physical Foundations",
        reasoningSkill: "Data-Based and Statistical Reasoning",
        timingBurden: 3
      },
      create: {
        slug: "electrochemistry",
        title: "Electrochemistry",
        section: "CP",
        summary: "Cell potential, redox balancing, and galvanic vs electrolytic logic.",
        highYield: true,
        contentMd: "Use Ecell = Ecathode - Eanode and sign conventions under pressure.",
        blueprintCategory: "Chemical and Physical Foundations",
        reasoningSkill: "Data-Based and Statistical Reasoning",
        timingBurden: 3
      }
    }),
    prisma.topic.upsert({
      where: { slug: "psych-social-theories" },
      update: {
        blueprintCategory: "Psychological, Social, and Biological Foundations",
        reasoningSkill: "Knowledge of Scientific Concepts and Principles",
        timingBurden: 2
      },
      create: {
        slug: "psych-social-theories",
        title: "Psych/Soc Theory Application",
        section: "PS",
        summary: "Distinguishing key frameworks and applying them to scenarios.",
        highYield: true,
        contentMd: "Contrast functionalism, conflict theory, symbolic interactionism, and social constructionism.",
        blueprintCategory: "Psychological, Social, and Biological Foundations",
        reasoningSkill: "Knowledge of Scientific Concepts and Principles",
        timingBurden: 2
      }
    })
  ]);

  await upsertQuestion(
    topics[0].id,
    "A competitive inhibitor is added to an enzyme reaction. Which change is expected?",
    3,
    "Competitive inhibition raises apparent Km with unchanged Vmax.",
    "Noncompetitive inhibition changes Vmax, not classic competitive inhibition.",
    "Identify inhibitor class before mapping graph effects.",
    [
      { label: "A", text: "Km increases, Vmax unchanged", isCorrect: true },
      { label: "B", text: "Km unchanged, Vmax decreases", isCorrect: false },
      { label: "C", text: "Km decreases, Vmax unchanged", isCorrect: false },
      { label: "D", text: "Km and Vmax both decrease", isCorrect: false }
    ]
  );

  await upsertQuestion(
    topics[0].id,
    "An enzyme-catalyzed reaction doubles in velocity when substrate concentration increases from 2 mM to 4 mM. Which statement is most likely true?",
    4,
    "At low substrate concentrations, velocity is approximately proportional to [S].",
    "Near Vmax, velocity no longer scales linearly with substrate concentration.",
    "Check whether substrate range sits well below or near Km before inferring kinetics.",
    [
      { label: "A", text: "The reaction is operating far below saturation", isCorrect: true },
      { label: "B", text: "The reaction is already at Vmax", isCorrect: false },
      { label: "C", text: "Km must be zero", isCorrect: false },
      { label: "D", text: "The enzyme is nonfunctional", isCorrect: false }
    ]
  );

  await upsertQuestion(
    topics[1].id,
    "In a CARS passage, the author presents an opposing view only to qualify it in the final paragraph. What is the likely function of that opposition?",
    3,
    "The opposition is used as a foil to sharpen the author's nuanced position.",
    "Many readers mistake mention of an opposing claim for endorsement.",
    "Track whether a claim is presented, endorsed, or qualified.",
    [
      { label: "A", text: "It is the thesis the author defends", isCorrect: false },
      { label: "B", text: "It is a foil used to refine the author’s final position", isCorrect: true },
      { label: "C", text: "It is irrelevant to the argument", isCorrect: false },
      { label: "D", text: "It proves the author is undecided", isCorrect: false }
    ]
  );

  await upsertQuestion(
    topics[1].id,
    "A passage describes industrialization as both liberating and destabilizing. Which answer best captures the author's tone?",
    4,
    "The tone is qualified: appreciative of benefits yet cautious about social costs.",
    "Extreme answer choices often ignore balanced language in the passage.",
    "Use qualifiers to eliminate absolutist tone choices.",
    [
      { label: "A", text: "Purely celebratory", isCorrect: false },
      { label: "B", text: "Uniformly pessimistic", isCorrect: false },
      { label: "C", text: "Qualified and ambivalent", isCorrect: true },
      { label: "D", text: "Detached and unrelated", isCorrect: false }
    ]
  );

  await upsertQuestion(
    topics[2].id,
    "For a galvanic cell, which relation correctly defines cell potential under standard conditions?",
    2,
    "E°cell = E°cathode − E°anode.",
    "Adding half-cell potentials without directionality gives incorrect sign.",
    "Identify oxidation and reduction first, then apply cathode-minus-anode.",
    [
      { label: "A", text: "E°cell = E°anode − E°cathode", isCorrect: false },
      { label: "B", text: "E°cell = E°cathode − E°anode", isCorrect: true },
      { label: "C", text: "E°cell = E°anode + E°cathode", isCorrect: false },
      { label: "D", text: "E°cell is always negative", isCorrect: false }
    ]
  );

  await upsertQuestion(
    topics[2].id,
    "If Q increases while temperature remains constant, what happens to Ecell for a spontaneous galvanic reaction?",
    3,
    "Ecell decreases as Q increases for a spontaneous reaction under the Nernst equation.",
    "Ignoring the logarithmic Q term leads to wrong direction predictions.",
    "Use Nernst qualitatively: larger Q reduces driving force.",
    [
      { label: "A", text: "Ecell increases", isCorrect: false },
      { label: "B", text: "Ecell stays constant", isCorrect: false },
      { label: "C", text: "Ecell decreases", isCorrect: true },
      { label: "D", text: "Sign cannot be inferred", isCorrect: false }
    ]
  );

  await upsertQuestion(
    topics[3].id,
    "A study shows stress responses vary with perceived control over events. Which framework is most directly being tested?",
    3,
    "Cognitive appraisal models emphasize perceived control and interpretation.",
    "Choosing broad social theories here misses the psychological construct being measured.",
    "Match measured variable to framework domain before selecting theory.",
    [
      { label: "A", text: "Cognitive appraisal theory", isCorrect: true },
      { label: "B", text: "Conflict theory", isCorrect: false },
      { label: "C", text: "Symbolic interactionism", isCorrect: false },
      { label: "D", text: "Exchange theory", isCorrect: false }
    ]
  );

  await upsertQuestion(
    topics[3].id,
    "A researcher finds behavior changes when participants know they are observed. Which bias is most relevant?",
    2,
    "The Hawthorne effect captures behavior changes due to awareness of observation.",
    "Confusing observer bias with participant reactivity changes interpretation of study validity.",
    "Separate who is biased: participant behavior vs observer measurement.",
    [
      { label: "A", text: "Selection bias", isCorrect: false },
      { label: "B", text: "Hawthorne effect", isCorrect: true },
      { label: "C", text: "Recall bias", isCorrect: false },
      { label: "D", text: "Confounding", isCorrect: false }
    ]
  );

  const passageExists = await prisma.passage.findFirst({ where: { title: "Art Criticism and Social Context" } });
  if (!passageExists) {
    await prisma.passage.create({
      data: {
        topicId: topics[1].id,
        title: "Art Criticism and Social Context",
        body: "The author contrasts aesthetic formalism with historically grounded criticism and argues for a blended interpretive method...",
        section: "CARS",
        questions: {
          create: [
            {
              prompt: "The author would most likely agree with which statement?",
              explanation: "The passage supports contextual interpretation over strict formalism.",
              indexInPassage: 1,
              answerChoices: {
                create: [
                  { label: "A", text: "Art can be interpreted apart from society", isCorrect: false },
                  { label: "B", text: "Context should inform interpretation", isCorrect: true },
                  { label: "C", text: "Only formal features matter", isCorrect: false },
                  { label: "D", text: "Historical context is irrelevant", isCorrect: false }
                ]
              }
            }
          ]
        }
      }
    });
  }

  const existingPlan = await prisma.studyPlan.findFirst({ where: { userId: user.id }, orderBy: { generatedAt: "desc" } });
  if (!existingPlan) {
    await prisma.studyPlan.create({
      data: {
        userId: user.id,
        weekStartDate: new Date("2026-03-30"),
        tasks: {
          create: [
            {
              title: "Timed CARS set: social sciences passages",
              section: "CARS",
              topic: "Argument inference",
              dueDate: new Date("2026-04-01"),
              durationMinutes: 90,
              priority: 5,
              status: "in_progress"
            },
            {
              title: "Biochem mixed block: enzymes and metabolism",
              section: "BB",
              topic: "Enzyme kinetics",
              dueDate: new Date("2026-04-02"),
              durationMinutes: 75,
              priority: 4
            },
            {
              title: "Physics review: circuits and power",
              section: "CP",
              topic: "Electric circuits",
              dueDate: new Date("2026-04-03"),
              durationMinutes: 60,
              priority: 4
            }
          ]
        }
      }
    });
  }

  await prisma.topicMastery.createMany({
    data: topics.map((topic, idx) => ({
      userId: user.id,
      topicId: topic.id,
      mastery: 55 + idx * 7,
      confidence: 50 + idx * 6
    })),
    skipDuplicates: true
  });

  await prisma.errorLogEntry.createMany({
    data: [
      {
        userId: user.id,
        section: "CARS",
        topic: "Author tone inference",
        mistakeType: "reasoning_error",
        note: "Anchored too hard on one extreme phrase and ignored paragraph contrast.",
        tags: ["cars", "inference", "tone"],
        recurrenceCount: 3
      },
      {
        userId: user.id,
        section: "BB",
        topic: "Inhibition type mapping",
        mistakeType: "content_gap",
        note: "Mixed up competitive vs noncompetitive graph changes under time pressure.",
        tags: ["biochem", "graphs"],
        recurrenceCount: 2
      }
    ],
    skipDuplicates: true
  });

  const convExists = await prisma.aiConversation.findFirst({ where: { userId: user.id, title: "CARS elimination workflow" } });
  if (!convExists) {
    const conv = await prisma.aiConversation.create({
      data: {
        userId: user.id,
        title: "CARS elimination workflow",
        mode: "cars_coach"
      }
    });

    await prisma.aiMessage.createMany({
      data: [
        {
          conversationId: conv.id,
          role: "user",
          content: "How can I avoid trap answers in CARS inference questions?"
        },
        {
          conversationId: conv.id,
          role: "assistant",
          content: "Anchor each choice to explicit passage evidence and eliminate absolute language unless the author is explicitly absolute."
        }
      ]
    });
  }

  console.log(`Seeded user ${user.email} with expanded MCAT content and diagnostics-ready data.`);
}

main()
  .catch((error) => {
    console.error(error);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
