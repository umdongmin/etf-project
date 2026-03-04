import os
from google import genai
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class LLMService:
    """최신 google-genai SDK를 활용한 전략 분석 서비스"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            # 기본 모델 설정
            self.model_name = 'gemini-2.5-flash'
        else:
            self.client = None
            self.model_name = None

    def generate_briefing(self, opt_results, baseline, period_text):
        """최적화 결과를 바탕으로 투자 인사이트 리포트 생성"""
        if not self.client:
            return "⚠️ Gemini API 키가 설정되지 않았습니다. 사이드바에서 설정해주세요."

        # 1. 상위 3개 전략 데이터 요약
        top_strats = opt_results[:3]
        strat_summary = ""
        for i, res in enumerate(top_strats):
            strat_summary += f"\n[전략 {i+1}]\n"
            strat_summary += f"- CAGR: {res['cagr']:.1f}%, MDD: {res['mdd']:.1f}%, Score: {res['score']:.2f}\n"
            strat_summary += f"- Sharpe: {res.get('sharpe', 0):.2f}, PF: {res.get('pf', 0):.2f}, Win Rate: {res.get('win_rate', 0):.1f}%\n"

        # 2. 프롬프트 구성
        prompt = f"""
        너는 20년 경력의 전문 퀀트 투자 전략가이자 리스크 관리자야. 
        사용자가 수행한 'TQQQ 골든 전략 최적화' 결과를 분석해서 핵심 인사이트 리포트를 작성해줘.

        [분석 대상 기간]: {period_text}
        [기준 성과 (Baseline)]: CAGR {baseline['cagr']:.1f}%, MDD {baseline['mdd']:.1f}%, Sharpe {baseline.get('sharpe', 0):.2f}
        [최적화된 상위 전략들]: {strat_summary}

        [리포트 작성 가이드]:
        1. **총평**: 현재 전략이 시장 대비 어느 정도의 경쟁력이 있는지 평가해줘.
        2. **장점 분석 (데이터 기반)**: 수익률 뿐만 아니라 샤프지수(변동성 대비 수익)와 손익비(PF) 관점에서 성과가 얼마나 견고해졌는지 수치로 분석해줘.
        3. **승률 및 리스크**: 승률과 최대 회복 기간 등을 고려할 때, 심리적으로 견디기 쉬운 전략인지 퀀트의 시각에서 조언해줘.
        4. **전략가로서의 제언**: 현재 찾은 최적 조합의 특징을 짚어주고, 향후 시장 변화 시 주의할 점을 알려줘.

        톤앤매너: 전문적이면서도 데이터에 근거한 신뢰감 있는 어조로, 친절하게 설명해줘. 한국어로 작성해줘.
        """

        try:
            # google-genai SDK 호출 방식 (최신 모델 gemini-2.5-flash 적용)
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text + f"\n\n(AI 모델: gemini-2.5-flash | SDK: google-genai)"
        except Exception as e:
            return f"❌ 리포트 생성 중 오류 발생: {str(e)}"

    @staticmethod
    def is_available():
        """API 키 설정 여부 확인"""
        return os.getenv("GEMINI_API_KEY") is not None
