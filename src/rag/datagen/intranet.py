"""가상 사내 인트라넷 카탈로그 — 실데이터 도착 전 리허설용 쌍둥이 데이터셋.

실제 대상 데이터는 "사내 사이트 검색"이다: 문서 하나 = 사이트의 페이지 하나로,
url(페이지 단위) + description + agent flow에 넘길 prompt + 수집 메타데이터
(담당자·버전·수집 시각)를 가진다. 이 모듈은 그 형태를 그대로 본뜬 가상 회사
"다온"의 인트라넷 카탈로그를 seed 고정으로 생성한다.

이 데이터셋의 존재 이유는 **양성 대조군(positive control)** 이다. korea.kr PoC는
base 모델이 이미 아는 공공 텍스트라 파인튜닝이 파고들 도메인 격차가 없었고, 그
결과 "개선 없음"이 데이터 탓인지 파이프라인 탓인지 구분할 수 없었다. 여기서는
격차를 설계로 주입한다:

  - 시스템마다 **사내 은어**(옛 시스템명, 팀 별명 — 예: 경비정산 '머니핀'을
    "아테나"라고 부름)를 두되, 은어는 corpus 본문 어디에도 등장하지 않는다
    (`generate`가 검증). 은어→시스템 연결은 오직 학습쌍(클릭로그 시뮬레이션)에만
    존재하므로, base 모델은 구조적으로 맞힐 수 없고 파인튜닝만 배울 수 있다.
  - 평가 쿼리는 `slice` 태그(standard | jargon)를 달고 나간다 — 전체 평균에
    묻히는 격차를 슬라이스별로 드러내기 위해서다.
  - 학습 쿼리와 평가 쿼리는 문장 템플릿 풀을 분리한다(표현 암기가 아니라
    어휘 연결을 배웠는지 측정).

corpus의 content(=임베딩 입력의 본문)는 description+본문이고, agent_prompt와
metadata는 임베딩에서 제외한다 — 식별자/운영 정보는 의미 검색의 신호가 아니라는
`core.formatting`의 원칙과 같다(합류 실험은 별도 과제).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

BASE_URL = "https://intra.daon.example"
COMPANY = "다온"


@dataclass(frozen=True)
class Task:
    name: str                  # 공식 업무명 — corpus 본문에 등장
    detail: str                # 가이드 절차 문장
    variant: str               # 구어 변형 — 평가 쿼리 전용(본문 비등장 표현)


@dataclass(frozen=True)
class System:
    slug: str
    name: str                  # 공식 시스템명 — corpus에 등장
    team: str
    desc: str                  # "~을/를 처리하는" 형태의 소개 절
    tasks: tuple[Task, ...]
    aliases: tuple[str, ...]   # 사내 은어 — corpus 등장 금지, 쿼리 전용


SYSTEMS: tuple[System, ...] = (
    System("hangyeol", "한결", "경영지원팀", "품의와 전자결재를 처리하는", (
        Task("품의서 상신", "결재 홈에서 새 품의서를 열고 결재선을 지정해 상신합니다.",
             "품의 올리기"),
        Task("휴가 결재 신청", "휴가 종류와 기간을 선택하면 부서장 결재로 자동 연결됩니다.",
             "연차 결재 올리는 거"),
        Task("지출 결의서 제출", "증빙을 첨부하고 계정 과목을 선택해 지출 결의를 제출합니다.",
             "지출결의 내는 법"),
    ), ("파피루스", "청기와")),
    System("peopleon", "피플온", "인사팀", "인사 정보와 증명서 발급을 담당하는", (
        Task("연차 잔여일 조회", "마이페이지의 근태 현황에서 남은 연차와 사용 이력을 확인합니다.",
             "남은 연차 확인"),
        Task("재직증명서 발급", "증명서 메뉴에서 용도를 입력하면 PDF로 즉시 발급됩니다.",
             "재직증명서 뽑기"),
        Task("급여명세서 확인", "급여 메뉴에서 월별 명세서를 열람하고 내려받을 수 있습니다.",
             "월급 명세서 보기"),
    ), ("세잎", "미리내")),
    System("moneypin", "머니핀", "재무팀", "법인카드와 경비 정산을 처리하는", (
        Task("법인카드 사용내역 정산", "카드 사용 건마다 사유와 프로젝트 코드를 입력해 정산합니다.",
             "법카 정산"),
        Task("개인경비 청구", "영수증을 촬영해 첨부하면 개인 경비로 청구할 수 있습니다.",
             "사비로 쓴 거 청구"),
        Task("출장비 정산", "출장 종료 후 교통·숙박 실비를 항목별로 등록해 정산합니다.",
             "출장 경비 처리"),
    ), ("아테나", "두꺼비")),
    System("gateone", "게이트원", "정보보안팀", "VPN과 원격접속을 제공하는", (
        Task("VPN 클라이언트 설치", "다운로드 센터에서 OS별 클라이언트를 받아 설치합니다.",
             "브이피엔 까는 법"),
        Task("재택 원격접속", "재택 근무 시 VPN 연결 후 사내망 자원에 접속합니다.",
             "집에서 사내망 접속"),
        Task("2차 인증 등록", "모바일 앱에서 OTP를 등록해야 첫 접속이 가능합니다.",
             "OTP 등록"),
    ), ("개미굴", "북극성")),
    System("notree", "노트리", "플랫폼개발팀", "사내 위키와 문서 협업을 담당하는", (
        Task("팀 스페이스 개설", "스페이스 만들기에서 팀 이름과 공개 범위를 정해 개설합니다.",
             "팀 위키 만들기"),
        Task("문서 버전 복원", "문서 이력 화면에서 원하는 시점의 버전으로 되돌립니다.",
             "예전 버전으로 되돌리기"),
        Task("문서 템플릿 적용", "회의록·기획서 등 표준 템플릿을 선택해 새 문서를 만듭니다.",
             "회의록 양식 쓰기"),
    ), ("코쿤", "감나무")),
    System("shipit", "쉽잇", "플랫폼개발팀", "서비스 배포 파이프라인을 관리하는", (
        Task("배포 승인 요청", "배포 계획을 등록하면 리뷰어 승인 후 파이프라인이 시작됩니다.",
             "릴리즈 승인 받기"),
        Task("롤백 실행", "배포 이력에서 직전 안정 버전을 선택해 롤백합니다.",
             "이전 버전으로 되돌리는 배포"),
        Task("파이프라인 실패 로그 확인", "실패한 단계를 클릭하면 상세 로그와 재시도 버튼이 나옵니다.",
             "배포 실패 원인 보기"),
    ), ("등대", "방아쇠")),
    System("argos", "아르고스", "SRE팀", "서비스 모니터링과 장애 알림을 담당하는", (
        Task("대시보드 생성", "메트릭을 골라 위젯을 배치하면 팀 대시보드가 만들어집니다.",
             "지표 보드 만들기"),
        Task("알림 규칙 설정", "임계값과 수신 채널을 정해 알림 규칙을 등록합니다.",
             "알람 조건 걸기"),
        Task("온콜 일정 확인", "온콜 캘린더에서 이번 주 당번과 에스컬레이션 순서를 확인합니다.",
             "이번주 당직 누구"),
    ), ("부엉이", "초롱불")),
    System("toksquare", "톡스퀘어", "커뮤니케이션팀", "사내 메신저와 채널 협업을 제공하는", (
        Task("채널 개설", "새 채널에서 공개 여부와 멤버를 정해 개설합니다.",
             "단톡방 만들기"),
        Task("외부 게스트 초대", "게스트 초대 메뉴에서 이메일로 외부 협력자를 초대합니다.",
             "외부인 초대"),
        Task("알림 음소거 설정", "채널별 알림을 시간대 단위로 음소거할 수 있습니다.",
             "알림 끄기"),
    ), ("모닥불", "제비")),
    System("roombook", "룸북", "총무팀", "회의실과 좌석 예약을 처리하는", (
        Task("회의실 예약", "층·시간대를 고르고 참석 인원을 입력해 회의실을 예약합니다.",
             "회의실 잡기"),
        Task("자율좌석 예약", "출근 전날부터 좌석 지도를 열어 자리를 선점할 수 있습니다.",
             "내일 자리 맡기"),
        Task("화상회의 장비 신청", "장비 신청 메뉴에서 카메라·스피커폰 대여를 신청합니다.",
             "화상 장비 빌리기"),
    ), ("둥지", "시루")),
    System("assethub", "애셋허브", "IT지원팀", "IT 자산과 라이선스를 관리하는", (
        Task("노트북 교체 신청", "사용 연한이 지난 장비는 교체 신청서로 새 장비를 받습니다.",
             "노트북 바꾸기"),
        Task("소프트웨어 라이선스 신청", "카탈로그에서 필요한 소프트웨어를 골라 신청합니다.",
             "유료 프로그램 설치 신청"),
        Task("자산 반납", "퇴사·부서 이동 시 보유 자산을 반납 등록합니다.",
             "장비 반납"),
    ), ("곳간", "낙타")),
    System("onepass", "원패스", "정보보안팀", "사내 계정과 SSO 인증을 담당하는", (
        Task("비밀번호 초기화", "본인 인증 후 새 비밀번호를 설정할 수 있습니다.",
             "비번 리셋"),
        Task("계정 잠금 해제", "로그인 5회 실패로 잠긴 계정을 본인 확인 후 해제합니다.",
             "계정 잠김 풀기"),
        Task("권한 그룹 변경", "부서 이동 시 소속 권한 그룹 변경을 신청합니다.",
             "권한 바꿔달라고 하기"),
    ), ("여의주", "은행나무")),
    System("helpme", "헬프미", "IT지원팀", "IT 장애 접수와 원격 지원을 담당하는", (
        Task("장애 티켓 접수", "증상과 스크린샷을 첨부해 장애 티켓을 접수합니다.",
             "고장 신고"),
        Task("원격 지원 요청", "상담원이 화면을 함께 보며 문제를 해결하는 원격 지원을 요청합니다.",
             "원격으로 봐달라고 하기"),
        Task("프린터 연결 설정", "층별 프린터 드라이버 설치와 연결 방법을 안내합니다.",
             "프린터 연결이 필요할 때"),
    ), ("풍금", "소라")),
    System("legalgate", "리걸게이트", "법무팀", "계약 검토와 법률 자문을 접수하는", (
        Task("계약서 검토 요청", "계약서 초안과 배경을 첨부해 검토를 요청합니다.",
             "계약서 봐달라고 하기"),
        Task("NDA 템플릿 내려받기", "표준 비밀유지계약서 템플릿을 내려받아 사용합니다.",
             "비밀유지 계약 양식"),
        Task("개인정보 처리 검토", "개인정보를 다루는 신규 기능은 사전 검토를 신청해야 합니다.",
             "개인정보 이슈 문의"),
    ), ("한산도", "미르")),
    System("baroBuy", "바로구매", "구매팀", "구매 요청과 거래처 관리를 처리하는", (
        Task("구매 품의 요청", "품목·수량·견적서를 첨부해 구매 품의를 올립니다.",
             "물품 사달라고 요청"),
        Task("거래처 등록", "신규 거래처의 사업자 정보와 통장 사본을 등록합니다.",
             "새 업체 등록"),
        Task("구매 진행 상태 조회", "내 요청 목록에서 발주·입고 단계를 확인합니다.",
             "주문 어디까지 왔나"),
    ), ("두레", "밤톨")),
    System("safecampus", "세이프캠퍼스", "정보보안팀", "보안 교육과 모의훈련을 운영하는", (
        Task("필수 보안교육 수강", "분기별 필수 과정을 기한 내에 수강해야 합니다.",
             "보안교육 듣기"),
        Task("이수증 발급", "수강 완료 과정의 이수증을 PDF로 발급합니다.",
             "수료증 뽑기"),
        Task("모의 피싱 훈련 결과 확인", "내 훈련 결과와 부서 평균을 비교해 볼 수 있습니다.",
             "피싱 테스트 결과"),
    ), ("반딧불", "옹달샘")),
    System("docspot", "독스팟", "경영지원팀", "공식 문서 중앙화와 보안 공유를 담당하는", (
        Task("부서 폴더 생성", "부서 관리자 권한으로 팀 문서 폴더를 만듭니다.",
             "팀 폴더 만들기"),
        Task("외부 공유 링크 발급", "만료일과 열람 범위를 정해 외부 공유 링크를 만듭니다.",
             "외부에 파일 보내기"),
        Task("문서 보안 등급 설정", "문서마다 대외비·사내한 등 보안 등급을 지정합니다.",
             "문서 대외비 걸기"),
    ), ("가람", "모래성")),
    System("dataon", "데이터온", "데이터팀", "사내 지표와 데이터셋을 제공하는", (
        Task("지표 대시보드 열람", "전사 KPI 대시보드는 데이터 포털 홈에서 바로 열람합니다.",
             "매출 지표 보는 곳"),
        Task("데이터셋 검색", "데이터 카탈로그에서 테이블 설명과 담당자를 검색합니다.",
             "테이블 찾기"),
        Task("쿼리 환경 신청", "분석용 SQL 실행 환경 계정을 신청합니다.",
             "SQL 돌릴 곳 신청"),
    ), ("우물", "노들")),
    System("recruita", "리크루타", "인사팀", "채용 공고와 면접 일정을 관리하는", (
        Task("사내 추천 등록", "추천할 지원자의 이력서를 등록하면 보상 대상이 됩니다.",
             "지인 추천하기"),
        Task("면접 일정 조율", "면접관 가능 시간을 등록하면 일정이 자동 조율됩니다.",
             "면접 시간 잡기"),
        Task("채용 공고 게시 요청", "부서 채용 계획을 입력해 공고 게시를 요청합니다.",
             "채용 공고 올리기"),
    ), ("솔개", "두견")),
    System("edubridge", "에듀브릿지", "인재개발팀", "직무 교육과 강의 수강을 운영하는", (
        Task("직무 교육 신청", "교육 카탈로그에서 과정을 골라 부서장 승인으로 신청합니다.",
             "강의 듣고 싶을 때"),
        Task("외부 교육비 지원 신청", "외부 강의·컨퍼런스는 사전 신청 시 비용이 지원됩니다.",
             "외부 강의 비용 지원"),
        Task("수강 이력 조회", "내 학습 페이지에서 이수 과정과 시간을 확인합니다.",
             "들은 강의 확인"),
    ), ("등불", "여울")),
    System("carepoint", "케어포인트", "인사팀", "복지포인트와 건강 복지를 담당하는", (
        Task("복지포인트 잔액 조회", "홈 화면에서 올해 포인트 잔액과 소멸 예정일을 확인합니다.",
             "복지포인트 얼마 남았나"),
        Task("건강검진 예약", "제휴 검진센터와 날짜를 골라 검진을 예약합니다.",
             "건강검진 날짜 잡기"),
        Task("단체보험 청구", "진료비 영수증을 첨부해 단체보험금을 청구합니다.",
             "보험금 청구"),
    ), ("보따리", "솜다리")),
    System("worklog", "워크로그", "인사팀", "출퇴근 기록과 근무제를 관리하는", (
        Task("출퇴근 기록 정정", "누락된 출퇴근 기록은 사유를 적어 정정 신청합니다.",
             "출근 찍는 걸 깜빡했을 때"),
        Task("재택근무 신청", "주간 단위로 재택 일자를 신청하고 부서장 승인을 받습니다.",
             "재택 신청"),
        Task("초과근무 신청", "야근·주말 근무는 사전 신청해야 수당이 정산됩니다.",
             "야근 등록"),
    ), ("자작나무", "댓돌")),
    System("issuego", "이슈고", "플랫폼개발팀", "프로젝트 이슈 트래킹을 제공하는", (
        Task("프로젝트 보드 생성", "새 프로젝트를 만들고 칸반 보드 컬럼을 구성합니다.",
             "칸반 보드 만들기"),
        Task("스프린트 설정", "기간과 목표를 정해 스프린트를 시작합니다.",
             "스프린트 돌리기"),
        Task("협력사 계정 초대", "외부 협력사 인원을 프로젝트 단위 게스트로 초대합니다.",
             "외주 인력 계정"),
    ), ("바둑판", "능금")),
    System("hubport", "허브포트", "플랫폼개발팀", "사내 API 게이트웨이를 운영하는", (
        Task("API 키 발급", "서비스를 등록하면 호출용 API 키가 발급됩니다.",
             "API 토큰 받기"),
        Task("호출량 한도 상향", "트래픽 증가가 예상되면 한도 상향을 신청합니다.",
             "쿼터 늘리기"),
        Task("API 스펙 문서 등록", "OpenAPI 스펙을 올리면 문서 페이지가 자동 생성됩니다.",
             "API 문서 올리기"),
    ), ("찻집", "팽나무")),
    System("brandbox", "브랜드박스", "디자인팀", "브랜드 자산과 디자인 요청을 관리하는", (
        Task("로고 파일 내려받기", "공식 로고의 다양한 포맷을 용도별로 내려받습니다.",
             "회사 로고 파일"),
        Task("발표자료 템플릿 사용", "대외 발표용 표준 슬라이드 템플릿을 제공합니다.",
             "PPT 양식"),
        Task("디자인 요청 접수", "배너·인쇄물 등 디자인 작업을 요청서로 접수합니다.",
             "디자인 의뢰"),
    ), ("물레", "댕기")),
    System("tripon", "트립온", "총무팀", "출장 신청과 예약을 처리하는", (
        Task("출장 신청", "목적지와 기간, 예상 경비를 입력해 출장을 신청합니다.",
             "출장 가려면"),
        Task("항공·숙박 예약", "승인된 출장은 제휴 예약 화면에서 항공권과 숙소를 예약합니다.",
             "비행기표 예약"),
        Task("출장 보고서 제출", "복귀 후 7일 이내에 결과 보고서를 제출합니다.",
             "출장 다녀와서 보고"),
    ), ("솔바람", "고드름")),
    System("mailguard", "메일가드", "정보보안팀", "이메일 보안과 스팸 차단을 담당하는", (
        Task("스팸 차단 해제", "격리된 정상 메일은 차단 해제를 요청해 수신합니다.",
             "메일이 스팸으로 빠질 때"),
        Task("대용량 첨부 발송", "2GB까지의 첨부는 대용량 발송 링크로 전송합니다.",
             "큰 파일 메일로 보내기"),
        Task("격리 메일 확인", "격리함에서 차단된 메일의 사유를 확인합니다.",
             "차단된 메일 보기"),
    ), ("골무", "능소화")),
    System("townhall", "타운홀", "커뮤니케이션팀", "전사 공지와 사내 소통을 담당하는", (
        Task("전사 공지 게시 요청", "부서 공지를 전사 채널에 게시하려면 요청서를 제출합니다.",
             "전체 공지 올리기"),
        Task("경영진 Q&A 제출", "분기 타운홀 미팅에 앞서 질문을 익명으로 제출합니다.",
             "경영진에게 질문"),
        Task("사내 설문 개설", "설문 빌더로 문항을 만들어 대상 조직에 발송합니다.",
             "설문조사 만들기"),
    ), ("꽃신", "서리꽃")),
)

# 시스템에 속하지 않는 전사 규정 페이지 — 자연 distractor + 표준 검색 대상.
POLICIES: tuple[tuple[str, str, str], ...] = (
    ("remote-work", "재택근무 규정",
     "주 2회까지 재택근무가 가능하며, 신청은 워크로그에서 주간 단위로 합니다. "
     "재택일에는 코어타임(10~16시) 접속 상태를 유지해야 합니다."),
    ("security-basics", "정보보안 기본 수칙",
     "사외 반출 문서는 보안 등급 확인 후 승인된 채널로만 공유합니다. "
     "화면 잠금과 비밀번호 규정 등 임직원 필수 보안 수칙을 안내합니다."),
    ("family-leave", "경조 휴가 및 경조금 규정",
     "결혼·출산·조사 등 경조사별 휴가 일수와 경조금 지급 기준을 안내합니다. "
     "신청은 증빙 서류를 첨부해 인사팀으로 접수합니다."),
    ("office-badge", "사무실 출입증 발급 안내",
     "신규 입사자와 분실 시 출입증 발급 절차를 안내합니다. "
     "임시 출입증은 안내데스크에서 당일 발급됩니다."),
    ("employee-discount", "임직원 제휴 할인 혜택",
     "통신·숙박·문화 제휴처의 임직원 할인 목록과 이용 방법을 안내합니다. "
     "제휴 코드는 분기마다 갱신됩니다."),
    ("privacy-internal", "개인정보 내부 관리 계획",
     "고객 개인정보 취급 부서의 접근 권한 원칙과 연 1회 점검 절차를 규정합니다. "
     "위반 사례는 정보보안팀에 즉시 신고해야 합니다."),
)

OWNERS = ("김서연", "박도윤", "이하람", "정시우", "최나린", "한지호", "오세아", "문가온")

PAGE_KINDS = ("home", "guide", "faq", "access", "notice", "release", "manual")


def _josa(word: str, with_batchim: str, without: str) -> str:
    """받침 유무로 조사를 고른다 (한글이 아니면 받침 없는 쪽)."""
    last = word[-1]
    if "가" <= last <= "힣" and (ord(last) - 0xAC00) % 28:
        return word + with_batchim
    return word + without


def _version(rng: random.Random) -> str:
    return f"v{rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 12)}"


def _metadata(rng: random.Random, version: str) -> dict:
    day = rng.randint(1, 28)
    month = rng.choice((5, 6, 7))
    return {
        "collected_by": rng.choice(OWNERS),
        "version": version,
        "collected_at": f"2026-{month:02d}-{day:02d}T{rng.randint(9, 18):02d}:00:00+09:00",
        "source": "intranet-crawler/0.4",
    }


def _page_body(sys: System, kind: str, version: str, rng: random.Random) -> tuple[str, str, str]:
    """(title, description, body) — kind별 어휘가 갈리도록 본문을 구성한다.

    같은 kind의 페이지끼리는 시스템명 말고는 거의 같은 어휘를 쓰게 되는데, 이는
    의도된 난이도다: 은어 쿼리("두꺼비 권한 신청")는 시스템명 연결 없이는 27개
    권한 페이지를 구분할 수 없다 — 대조군이 노리는 바로 그 지점.
    """
    name, team = sys.name, sys.team
    tasks = sys.tasks
    if kind == "home":
        title = f"{name} — {sys.desc} 사내 시스템"
        desc = f"{_josa(name, '은', '는')} {sys.desc} 사내 시스템입니다. {team}에서 운영합니다."
        body = (
            f"{desc}\n주요 기능: {', '.join(t.name for t in tasks)}.\n"
            f"전 임직원이 기본 열람할 수 있으며, 세부 절차는 사용 가이드 페이지를 참고하세요. "
            f"운영 문의: {team}"
        )
    elif kind == "guide":
        title = f"{name} 사용 가이드"
        desc = f"{name}의 주요 업무 절차를 단계별로 안내합니다."
        steps = "\n".join(f"{i + 1}. {t.name}: {t.detail}" for i, t in enumerate(tasks))
        body = f"{desc}\n{steps}\n절차가 바뀐 경우 이 페이지가 우선합니다."
    elif kind == "faq":
        title = f"{name} 자주 묻는 질문"
        desc = f"{name} 사용 중 자주 겪는 문제와 해결 방법을 모았습니다."
        qa = "\n".join(
            f"Q. {t.name} 진행 중 오류가 나요.\n"
            f"A. {t.detail} 같은 오류가 반복되면 {team}에 문의하세요."
            for t in tasks[:2]
        )
        body = (
            f"{desc}\n{qa}\nQ. 로그인이 안 돼요.\n"
            f"A. {_josa(name, '은', '는')} 사내 SSO 계정으로 로그인합니다. "
            f"비밀번호·계정 잠금 문제는 계정 포털에서 해결하세요."
        )
    elif kind == "access":
        title = f"{name} 접근 권한·계정 안내"
        desc = f"{name}의 이용 권한 체계와 권한 신청 절차를 안내합니다."
        body = (
            f"{desc}\n기본 열람 권한은 전 임직원에게 부여되어 있습니다. "
            f"{tasks[0].name} 등 처리 작업에는 담당 권한 그룹이 필요하며, "
            f"권한 신청 메뉴에서 요청 사유를 적어 제출하면 {team} 승인 후 부여됩니다. "
            f"협력사 등 외부 사용자 계정은 보안 검토 후 발급됩니다."
        )
    elif kind == "notice":
        week = rng.choice(("첫", "둘", "셋"))
        day = rng.choice(("화", "수", "목"))
        title = f"{name} 점검·장애 공지"
        desc = f"{name}의 정기 점검 일정과 장애 이력을 공지합니다."
        body = (
            f"{desc}\n정기 점검: 매월 {week}째 주 {day}요일 22:00~24:00 "
            f"(점검 중 서비스가 일시 중단될 수 있습니다).\n"
            f"최근 공지: {version} 배포에 따른 사전 점검을 안내드립니다."
        )
    elif kind == "release":
        picked = rng.choice(tasks)
        title = f"{name} 릴리스 노트 {version}"
        desc = f"{name}의 최신 배포 버전 변경 사항입니다."
        body = (
            f"{desc}\n{version} 주요 변경: {picked.name} 화면 개선, 처리 속도 향상, "
            f"접근성 보완.\n이전 버전 기록은 페이지 하단 아카이브에서 확인할 수 있습니다."
        )
    else:  # manual
        title = f"{name} 관리자 매뉴얼"
        desc = f"{name} 운영 담당자를 위한 관리 기능 설명서입니다."
        body = (
            f"{desc}\n구성원 권한 일괄 관리, 감사 로그 조회, 공지 배너 설정 방법을 다룹니다. "
            f"관리자 메뉴는 {team}이 지정한 운영 담당자에게만 노출됩니다."
        )
    return title, desc, body


def _agent_prompt(sys: System, kind: str, url: str) -> str:
    """이 페이지를 agent flow에 물릴 때 함께 전달되는 프롬프트 (임베딩 제외)."""
    duties = {
        "home": "시스템이 무엇을 하는 곳인지 소개하고 알맞은 하위 페이지로 안내하세요.",
        "guide": "사용자의 요청을 아래 업무 중 하나로 분류해 절차를 단계별로 안내하세요: "
                 + ", ".join(t.name for t in sys.tasks) + ".",
        "faq": "증상을 먼저 확인하고 해당하는 문답을 찾아 해결 순서를 안내하세요.",
        "access": "필요한 권한 수준을 확인하고 권한 신청 절차를 안내하세요.",
        "notice": "점검 일정과 최근 공지를 사실대로 전달하세요.",
        "release": "버전별 변경 사항을 요약해 전달하세요.",
        "manual": "운영 담당자인지 확인한 뒤 관리 기능을 안내하세요.",
    }[kind]
    return (
        f"당신은 {COMPANY} 인트라넷의 '{sys.name}' 안내 에이전트입니다. "
        f"이 페이지({url})의 내용만 근거로 답하세요. {duties} "
        f"페이지에 없는 내용은 지어내지 말고 {sys.team} 문의를 안내하세요."
    )


def build_pages(rng: random.Random) -> list[dict]:
    """카탈로그 전체 페이지 — corpus.jsonl 레코드(풍부한 필드 포함)."""
    pages: list[dict] = []
    for sys in SYSTEMS:
        version = _version(rng)
        for kind in PAGE_KINDS:
            url = f"{BASE_URL}/{sys.slug}/{kind}"
            title, desc, body = _page_body(sys, kind, version, rng)
            pages.append({
                "url": url,
                "title": title,
                "description": desc,
                "content": body,
                "agent_prompt": _agent_prompt(sys, kind, url),
                "metadata": _metadata(rng, version),
                "system": sys.slug,
                "kind": kind,
            })
    for slug, title, body in POLICIES:
        url = f"{BASE_URL}/policy/{slug}"
        desc = body.split(". ")[0] + "."
        pages.append({
            "url": url,
            "title": title,
            "description": desc,
            "content": body,
            "agent_prompt": (
                f"당신은 {COMPANY} 인트라넷의 사내 규정 안내 에이전트입니다. "
                f"이 페이지({url})의 규정 내용만 근거로 답하고, 해석이 갈리는 사안은 "
                f"담당 부서 문의를 안내하세요."
            ),
            "metadata": _metadata(rng, "v1.0.0"),
            "system": "policy",
            "kind": slug,
        })
    return pages


# ── 쿼리 템플릿 — (템플릿, 정답 페이지 kind) ─────────────────────────────────
# 학습(클릭로그 시뮬레이션)과 평가는 풀을 분리한다: 평가가 학습 문장의 암기가
# 아니라 어휘 연결(은어→시스템, 구어→업무)의 일반화를 재도록.

STANDARD_TRAIN = (
    ("{name} {task} 방법", "guide"),
    ("{name}에서 {task} 하는 법", "guide"),
    ("{name} {task} 어떻게 하나요", "guide"),
    ("{name} 권한 신청", "access"),
    ("{name} 점검 일정", "notice"),
    ("{name} 새 기능", "release"),
    ("{name} 오류 문의", "faq"),
)
STANDARD_EVAL = (
    ("{name} {task} 절차", "guide"),
    ("{name}에서 {task} 하려면 어디로 가", "guide"),
    ("{name} 이용 권한 요청하고 싶어요", "access"),
    ("{name} 언제 점검해", "notice"),
)
JARGON_TRAIN = (
    ("{alias} {task} 방법", "guide"),
    ("{alias}에서 {task} 어떻게 해", "guide"),
    ("{alias} 권한 신청", "access"),
    ("{alias} 들어가는 법", "home"),
    ("{alias} 오류 나요", "faq"),
)
JARGON_EVAL = (
    ("{alias} {task} 어디서 해", "guide"),
    ("{alias} 계정 권한 부탁해요", "access"),
    ("{alias} 안 열려요", "faq"),
    ("{alias} 요즘 어디서 봐", "home"),
)


def _pair(query: str, page: dict, slice_name: str) -> dict:
    return {
        "query": query,
        "positive": {"title": page["title"], "content": page["content"]},
        "slice": slice_name,
    }


def build_queries(pages: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """(train_pairs, eval_pairs) — 클릭로그 시뮬레이션과 슬라이스 태그 평가 쿼리."""
    by_key = {(p["system"], p["kind"]): p for p in pages}
    train: list[dict] = []
    eval_pairs: list[dict] = []

    for sys in SYSTEMS:
        def page(kind: str) -> dict:
            return by_key[(sys.slug, kind)]  # noqa: B023 — 루프 내 즉시 사용

        # 표준 학습: 업무명은 공식명으로 (실로그의 다수 트래픽에 해당)
        for template, kind in STANDARD_TRAIN:
            if "{task}" in template:
                for task in sys.tasks:
                    train.append(_pair(
                        template.format(name=sys.name, task=task.name), page(kind), "standard"
                    ))
            else:
                train.append(_pair(template.format(name=sys.name), page(kind), "standard"))

        # 은어 학습: 두 은어 모두, 업무는 앞의 2개만 — 평가에서 세 번째 업무로 일반화 확인
        for alias in sys.aliases:
            for template, kind in JARGON_TRAIN:
                if "{task}" in template:
                    for task in sys.tasks[:2]:
                        train.append(_pair(
                            template.format(alias=alias, task=task.name), page(kind), "jargon"
                        ))
                else:
                    train.append(_pair(template.format(alias=alias), page(kind), "jargon"))

        # 표준 평가: 구어 변형(variant)으로 — 본문 비등장 표현의 일반화 측정
        picked = rng.sample(list(sys.tasks), 2)
        for (template, kind), task in zip(STANDARD_EVAL[:2], picked):
            eval_pairs.append(_pair(
                template.format(name=sys.name, task=task.variant), page(kind), "standard"
            ))
        for template, kind in STANDARD_EVAL[2:]:
            eval_pairs.append(_pair(template.format(name=sys.name), page(kind), "standard"))

        # 은어 평가: 학습에 안 나온 템플릿 + 세 번째 업무의 구어 변형
        third = sys.tasks[2] if len(sys.tasks) > 2 else sys.tasks[-1]
        alias_a, alias_b = sys.aliases[0], sys.aliases[-1]
        eval_pairs.append(_pair(
            JARGON_EVAL[0][0].format(alias=alias_a, task=third.variant),
            page(JARGON_EVAL[0][1]), "jargon",
        ))
        eval_pairs.append(_pair(
            JARGON_EVAL[1][0].format(alias=alias_b), page(JARGON_EVAL[1][1]), "jargon"
        ))
        eval_pairs.append(_pair(
            JARGON_EVAL[2][0].format(alias=alias_a), page(JARGON_EVAL[2][1]), "jargon"
        ))
        eval_pairs.append(_pair(
            JARGON_EVAL[3][0].format(alias=alias_b), page(JARGON_EVAL[3][1]), "jargon"
        ))

    # 규정 페이지: 표준 쿼리만 (은어 없음)
    for slug, title, _body in POLICIES:
        page = by_key[("policy", slug)]
        topic = title.split(" 규정")[0].split(" 안내")[0]
        train.append(_pair(f"{topic} 규정", page, "standard"))
        train.append(_pair(f"{title} 어디서 봐요", page, "standard"))
        eval_pairs.append(_pair(f"{topic} 기준이 궁금해요", page, "standard"))

    rng.shuffle(train)
    return train, eval_pairs


def generate(seed: int = 20260718) -> tuple[list[dict], list[dict], list[dict]]:
    """(corpus_pages, train_pairs, eval_pairs) — seed 고정, 생성 후 불변 조건 검증.

    불변 조건: ① 은어는 corpus 어떤 텍스트 필드에도 등장하지 않는다(양성 대조군의
    전제 — 깨지면 base도 맞힐 수 있게 되어 실험이 무효). ② 평가 쿼리 문자열은
    학습 쿼리와 겹치지 않는다(암기 측정 방지).
    """
    rng = random.Random(seed)
    pages = build_pages(rng)
    train, eval_pairs = build_queries(pages, rng)

    aliases = [a for sys in SYSTEMS for a in sys.aliases]
    for p in pages:
        searchable = " ".join((p["title"], p["description"], p["content"]))
        for alias in aliases:
            if alias in searchable:
                raise ValueError(f"은어 '{alias}'가 corpus에 등장합니다: {p['url']}")

    train_queries = {r["query"] for r in train}
    overlap = train_queries & {r["query"] for r in eval_pairs}
    if overlap:
        raise ValueError(f"학습/평가 쿼리 중복: {sorted(overlap)[:5]}")
    return pages, train, eval_pairs
