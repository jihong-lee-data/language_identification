# Language Identification API

### 라이브러리 설치

```bash
pip install -r requrirements.txt
```

### 모델 다운로드

```
추후 작성
```

### API 실행

```bash
uvicorn app.main:app --host x.x.x.x --port xxxx
```

### API 사용

### `POST /api/langid/predict`

단일 텍스트에 대한 언어 감지 결과를 반환합니다.

- **Request**
  - `text`: string: 텍스트 입력
  - `n`: int: 최대 언어 수 (확률 순, default 1)
  - 예시

    ```json
    {
        "text": "안녕하세요. 이것은 예시입니다.",
        "n": 3
    }
    ```

- **Response**
  - `pred`: key: 언어 코드(ISO 639-1)
  - `prob`: value: 언어 확률(float)
  - 예시

    ```json
    {
        "ko": 1.0,
        "id": 2.3814326510773753e-08,
        "it": 7.7582207325122e-09
    }
    ```

### `POST /api/langid/predictb`

다중 텍스트에 대한 언어 감지 결과를 반환합니다.

- **Request**
  - `text`: list of string: 텍스트 입력 리스트
  - `n`: int: 최대 언어 수 (확률 순, default 1)
  - 예시
  
    ```json
    {
        "text": [
            "안녕하세요. 이것은 예시입니다.",
            "こんにちは。これは例です。",
            "Hello. This is an example."
        ],
        "n": 2
    }
    ```

- **Response**
  - `result`: list of list of dict: 언어 감지 결과 리스트 (2차원 리스트)
  - 예시

    ```json
        {
        "result": [
            [
                {
                    "prediction": "ko",
                    "probability": 1.0,
                    "rank": 1,
                    "english": "Korean",
                    "korean": "한국어"
                },
                {
                    "prediction": "id",
                    "probability": 2.3814326510773753e-08,
                    "rank": 2,
                    "english": "Indonesian",
                    "korean": "인도네시아어"
                }
            ],
            [
                {
                    "prediction": "ja",
                    "probability": 1.0,
                    "rank": 1,
                    "english": "Japanese",
                    "korean": "일본어"
                },
                {
                    "prediction": "sw",
                    "probability": 8.151616270879458e-09,
                    "rank": 2,
                    "english": "Swahili",
                    "korean": "스와힐리어"
                }
            ],
            [
                {
                    "prediction": "en",
                    "probability": 0.9999998807907104,
                    "rank": 1,
                    "english": "English",
                    "korean": "영어"
                },
                {
                    "prediction": "id",
                    "probability": 8.273090656985005e-08,
                    "rank": 2,
                    "english": "Indonesian",
                    "korean": "인도네시아어"
                }
            ]
        ]
    }
    ```

### `POST /api/iso/search`

ISO 코드 테이블에서 쿼리를 검색한 결과를 반환합니다.

- **Request**
  - `query`: string: 검색할 쿼리
  - `tol`: int: 허용 오차 (default 2)

    - 0 - exact match
    - 1 - case-insensitive match
    - 2 - character-wise match

  - `code`: string: 검색할 테이블 필드(default "" - 전체 필드)

    - e - 영어명
    - k - 한국어명
    - 1 - ISO-639-1
    - 2 - ISO-639-2

        복수 필드 가능(e.g., 'ek1')

  - 예시

    ```json
    {
        "query": "영어",
        "tol": 2,
        "code": "k1"
    }
    ```

- **Response**
  - `result`: list of dict: 검색 결과 리스트
  - 예시

    ```json
    {
        "result": [
            {
                "English": "English",
                "Korean": "영어",
                "ISO 639-1 Code": "en",
                "ISO 639-2 Code": "eng"
            }
        ]
    }
    ```

### `POST /api/iso/convert`

주어진 텍스트를 다른 언어 코드로 변환한 결과를 반환합니다.

- **Request**
  - `src`: string: 변환할 텍스트
  - `src_code`: string: 변환할 언어 코드의 필드, 복수 필드 허용 (default "" - 전체 필드에서 검색)
  - `dst_code`: string: 변환될 언어 코드 (default "1")
  - 예시

    ```json
    {
        "src": "en",
        "src_code": "",
        "dst_code": "k"
    }

    ```

- **Response**
  - `result`: string: 변환된 코드
  - 예시

    ```json
    {
        "result": "영어"
    }
    ```
