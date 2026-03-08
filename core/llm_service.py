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

    def generate_rolling_report(self, rolling_results, summary_metrics, window_years):
        """롤링 윈도우 분석 결과를 바탕으로 전략 강건성 진단 리포트 생성"""
        if not self.client:
            return "⚠️ Gemini API 키가 설정되지 않았습니다. 사이드바에서 설정해주세요."

        # 1. 롤링 결과 데이터 요약 (상위 3개, 하위 3개 구간 추출)
        sorted_res = sorted(rolling_results, key=lambda x: x['raw_ret'], reverse=True)
        top_3 = sorted_res[:3]
        bottom_3 = sorted_res[-3:]

        rolling_summary = f"""
        [전체 통계]
        - 평균 수익률: {summary_metrics['avg_ret']:.1f}%
        - 평균 CAGR: {summary_metrics['avg_cagr']:.1f}%
        - 수익률 표준편차: {summary_metrics['std_ret']:.1f}%
        - 평균 MDD: {summary_metrics['avg_mdd']:.1f}%
        - 수익 구간 확률: {summary_metrics['win_rate']:.1f}%

        [베스트 구간 (Best 3)]
        """
        for r in top_3:
            rolling_summary += f"- {r['구간']}: 수익률 {r['최종 수익률']}, CAGR {r['연평균(CAGR)']}, MDD {r['MDD']}\n"
        
        rolling_summary += "\n[워스트 구간 (Worst 3)]\n"
        for r in bottom_3:
            rolling_summary += f"- {r['구간']}: 수익률 {r['최종 수익률']}, CAGR {r['연평균(CAGR)']}, MDD {r['MDD']}\n"

        # 2. 프롬프트 구성
        prompt = f"""
        너는 20년 경력의 전문 퀀트 투자 전략가이자 리스크 관리 전문 금융 공학자야. 
        사용자가 수행한 '{window_years}년 단위 롤링 윈도우 분석' 결과를 정밀 진단해서 리포트를 작성해줘.

        [분석 결과 데이터]:
        {rolling_summary}

        [리포트 작성 가이드]:
        1. **강건성(Robustness) 평가**: 수익률 표준편차와 수익 구간 확률을 바탕으로, 이 전략이 특정 시점의 운에 의존하는지 아니면 어떤 장세에서도 통하는 시스템인지 평가해줘.
        2. **리스크 진단**: 워스트 구간에서의 CAGR과 MDD를 분석하여, 투자자가 가장 고통스러운 시기에 어느 정도의 자산 감소를 견뎌야 하는지 조언해줘.
        3. **통계적 신뢰도**: 평균 수익률 대비 표준편차의 비율(변동성)을 고려할 때, 미래에도 이와 비슷한 성과가 재현될 가능성이 얼마나 높은지 분석해줘.
        4. **전략적 제안**: 현재의 롤링 성과를 더 안정적으로 개선하기 위해(예: 표준편차 낮추기) 보완할 점이 있다면 제안해줘.

        톤앤매너: 데이터에 근거하여 냉철하면서도 투자자에게 실질적인 도움이 되는 전문적인 어조로 작성해줘. 한국어로 작성해줘.
        """

        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text + f"\n\n(AI 모델: gemini-2.5-flash | 롤링 분석 전용)"
        except Exception as e:
            return f"❌ 리포트 생성 중 오류 발생: {str(e)}"

    @staticmethod
    def is_available():
        """API 키 설정 여부 확인"""
        return os.getenv("GEMINI_API_KEY") is not None
