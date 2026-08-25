-- CreateEnum
CREATE TYPE "SessionKind" AS ENUM ('diagnostic', 'question', 'passage', 'full_length');

-- CreateEnum
CREATE TYPE "SessionStatus" AS ENUM ('created', 'in_progress', 'submitted');

-- AlterEnum
ALTER TYPE "TutorMode" ADD VALUE 'plan';

-- AlterTable
ALTER TABLE "Topic" ADD COLUMN     "blueprintCategory" TEXT,
ADD COLUMN     "reasoningSkill" TEXT,
ADD COLUMN     "timingBurden" INTEGER NOT NULL DEFAULT 3;

-- CreateTable
CREATE TABLE "PracticeSession" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "kind" "SessionKind" NOT NULL,
    "section" "McatSection",
    "timed" BOOLEAN NOT NULL DEFAULT true,
    "status" "SessionStatus" NOT NULL DEFAULT 'created',
    "startedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "submittedAt" TIMESTAMP(3),
    "totalItems" INTEGER NOT NULL,
    "totalCorrect" INTEGER NOT NULL DEFAULT 0,
    "totalElapsedSeconds" INTEGER NOT NULL DEFAULT 0,
    "scorePercent" INTEGER,
    "contextNote" TEXT,

    CONSTRAINT "PracticeSession_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "PracticeSessionItem" (
    "id" TEXT NOT NULL,
    "sessionId" TEXT NOT NULL,
    "orderIndex" INTEGER NOT NULL,
    "section" "McatSection" NOT NULL,
    "topicId" TEXT,
    "questionId" TEXT,
    "passageQuestionId" TEXT,
    "blueprintCategory" TEXT,
    "reasoningSkill" TEXT,

    CONSTRAINT "PracticeSessionItem_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "PracticeAnswer" (
    "id" TEXT NOT NULL,
    "sessionId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "sessionItemId" TEXT NOT NULL,
    "section" "McatSection" NOT NULL,
    "questionId" TEXT,
    "passageQuestionId" TEXT,
    "selectedChoiceId" TEXT NOT NULL,
    "isCorrect" BOOLEAN NOT NULL,
    "confidence" INTEGER NOT NULL,
    "elapsedSeconds" INTEGER NOT NULL,
    "mistakeType" "MistakeType",
    "missClass" TEXT,
    "blueprintCategory" TEXT,
    "reasoningSkill" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "PracticeAnswer_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "PracticeSession_userId_kind_startedAt_idx" ON "PracticeSession"("userId", "kind", "startedAt");

-- CreateIndex
CREATE INDEX "PracticeSessionItem_sessionId_orderIndex_idx" ON "PracticeSessionItem"("sessionId", "orderIndex");

-- CreateIndex
CREATE INDEX "PracticeAnswer_sessionId_userId_idx" ON "PracticeAnswer"("sessionId", "userId");

-- CreateIndex
CREATE INDEX "PracticeAnswer_userId_createdAt_idx" ON "PracticeAnswer"("userId", "createdAt");

-- AddForeignKey
ALTER TABLE "PracticeSession" ADD CONSTRAINT "PracticeSession_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PracticeSessionItem" ADD CONSTRAINT "PracticeSessionItem_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "PracticeSession"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PracticeSessionItem" ADD CONSTRAINT "PracticeSessionItem_topicId_fkey" FOREIGN KEY ("topicId") REFERENCES "Topic"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PracticeSessionItem" ADD CONSTRAINT "PracticeSessionItem_questionId_fkey" FOREIGN KEY ("questionId") REFERENCES "Question"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PracticeSessionItem" ADD CONSTRAINT "PracticeSessionItem_passageQuestionId_fkey" FOREIGN KEY ("passageQuestionId") REFERENCES "PassageQuestion"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PracticeAnswer" ADD CONSTRAINT "PracticeAnswer_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "PracticeSession"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PracticeAnswer" ADD CONSTRAINT "PracticeAnswer_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "PracticeAnswer" ADD CONSTRAINT "PracticeAnswer_sessionItemId_fkey" FOREIGN KEY ("sessionItemId") REFERENCES "PracticeSessionItem"("id") ON DELETE CASCADE ON UPDATE CASCADE;
