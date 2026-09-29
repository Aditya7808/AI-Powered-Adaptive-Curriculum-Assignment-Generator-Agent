from src.agents.adaptation_agent import adapt_curriculum
from src.agents.assignment_agent import generate_assignment
from src.agents.curriculum_agent import generate_curriculum
from src.agents.evaluation_agent import analyze_performance
from src.agents.profiling_agent import LearnerProfilingAgent
from src.agents.rag_agent import build_rag_graph
from src.documents.ingestion import DocumentIngestionService
from src.documents.vectorstore import VectorStore
from src.retrieval.embeddings import EmbeddingService
from src.schemas.assignment import Assignment, DifficultyLevel
from src.schemas.curriculum import Curriculum
from src.schemas.documents import UploadedDocument
from src.schemas.learner import LearnerProfile, LearningObjective, SkillGapReport
from src.schemas.performance import (
    AdjustmentRecommendation,
    PerformanceReport,
    SubmissionResult,
)
from src.schemas.rag import RAGAnswer
from src.storage import db


class AdaptiveLearningOrchestrator:
    def __init__(self):
        db.init_db()
        self.rag_graph = build_rag_graph()
        self.profiler = LearnerProfilingAgent()

    def upload_documents(self, files: list[str]) -> list[UploadedDocument]:
        emb = EmbeddingService()
        vs = VectorStore()
        service = DocumentIngestionService(emb, vs)
        results = []
        for f in files:
            doc, _ = service.ingest(f)
            results.append(doc)
        return results

    def list_documents(self) -> list[UploadedDocument]:
        return db.get_all_documents()

    def delete_document(self, doc_id: str) -> None:
        emb = EmbeddingService()
        vs = VectorStore()
        service = DocumentIngestionService(emb, vs)
        service.delete(doc_id)

    def ask_documents(
        self, question: str, doc_ids: list[str] | None = None
    ) -> RAGAnswer:
        from src.rag.knowledge_service import KnowledgeService

        ks = KnowledgeService()
        return ks.ask(question, doc_ids)

    def onboard_learner(
        self, profile: LearnerProfile, objectives: list[LearningObjective]
    ) -> SkillGapReport:
        return self.profiler.analyze(profile, objectives)

    def build_curriculum(self, profile: LearnerProfile, objectives: str) -> Curriculum:
        curr = generate_curriculum(profile, objectives)
        # Store in DB...
        return curr

    def create_assignment(
        self,
        curriculum: Curriculum,
        module_id: str,
        difficulty: DifficultyLevel,
        question_mix: dict,
        learner: LearnerProfile,
    ) -> Assignment:
        module = next(m for m in curriculum.modules if m.module_id == module_id)
        return generate_assignment(module, difficulty, question_mix, learner)

    def record_submission(self, submission: SubmissionResult) -> None:
        pass  # Store to DB

    def analyze_and_adapt(
        self,
        learner: LearnerProfile,
        submissions: list[SubmissionResult],
        assignments: list[Assignment],
        curriculum: Curriculum,
    ) -> tuple[PerformanceReport, list[AdjustmentRecommendation], Curriculum | None]:
        report, recs = analyze_performance(
            learner.learner_id, submissions, assignments, curriculum
        )
        new_curr = None
        if recs:
            new_curr = adapt_curriculum(curriculum, learner, recs)
        return report, recs, new_curr
