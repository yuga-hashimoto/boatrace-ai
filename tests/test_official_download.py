from boatrace_ai.collect.official_download import parse_program_text, parse_result_text


PROGRAM_SAMPLE = """STARTB
19BBGN
ボートレース下　関   　３月１０日  ＭＮＢＲ下関１２ｔｈ  第　４日

                            ＊＊＊　番組表　＊＊＊

          ＭＮＢＲ下関１２ｔｈ　巌流本舗杯　　　　　　　　　

   第　４日          ２０２５年　３月１０日                  ボートレース下　関

　１Ｒ  一般　　　　          Ｈ１８００ｍ  電話投票締切予定１７：２２ 
-------------------------------------------------------------------------------
艇 選手 選手  年 支 体級    全国      当地     モーター   ボート   今節成績  早
番 登番  名   齢 部 重別 勝率  2率  勝率  2率  NO  2率  NO  2率  １２３４５６見
-------------------------------------------------------------------------------
1 3788一宮稔弘53徳島53B2 5.46 35.44 6.17 50.00 53 34.70 59 29.53 434 22       9
2 4753森　照夫36福岡56B1 4.72 25.93 4.32 21.05 61 40.45 14 31.60 4 244        6
3 5012加倉侑征27福岡53B1 5.62 37.89 4.25 25.00 52 31.43 46 39.13 322 52      10
"""


RESULT_SAMPLE = """STARTK
19KBGN
下　関［成績］      3/10      ＭＮＢＲ下関１２ｔｈ  第 4日

                            ＊＊＊　成績　＊＊＊

          ＭＮＢＲ下関１２ｔｈ　巌流本舗杯　　　　　　　　　

   第 4日          2025/ 3/10                             ボートレース下　関

   1R       一般　　　　                 H1800m  曇り  風  南　　 2m  波　  2cm
  着 艇 登番 　選　手　名　　ﾓｰﾀｰ ﾎﾞｰﾄ 展示 進入 ｽﾀｰﾄﾀｲﾐﾝｸ ﾚｰｽﾀｲﾑ 逃げ　　　
-------------------------------------------------------------------------------
  01  1 3788 一　宮　　稔　弘 53   59  6.71   1    0.07     2.03.6
  02  4 5029 中　　　　亮　太 35   13  6.67   4    0.05     2.05.5
  03  5 4871 菊　池　　宏　志 22   27  6.71   5    0.09     2.09.0
  S2  2 4753 森　　　　照　夫 61   14  6.70   2    0.12      .  .

        単勝     1          150
        ３連単   1-4-5     1970  人気     7
"""


def test_parse_program_text_extracts_card_fields():
    records = parse_program_text(PROGRAM_SAMPLE)

    record = records[("下関", 1)]
    assert record["meeting_name"] == "MNBR下関12th 巌流本舗杯"
    assert record["deadline"] == "17:22"
    assert record["card"]["entrants"][0]["racer_id"] == "3788"
    assert record["card"]["entrants"][0]["name"] == "一宮稔弘"
    assert record["card"]["entrants"][0]["motor_no"] == 53


def test_parse_result_text_extracts_finish_weather_and_payouts():
    records = parse_result_text(RESULT_SAMPLE)

    record = records[("下関", 1)]
    result = record["result"]
    assert record["meeting_name"] == "MNBR下関12th 巌流本舗杯"
    assert result["technique"] == "逃げ"
    assert result["weather"]["weather"] == "曇り"
    assert result["weather"]["wind_speed_mps"] == 2.0
    assert result["entrants"][0]["finish_position"] == 1
    assert result["start_timings"]["1"] == 0.07
    assert result["payouts"][0]["bet_type"] == "単勝"
    assert result["payouts"][1]["bet_type"] == "3連単"
    assert result["payouts"][1]["combination"] == "1-4-5"
