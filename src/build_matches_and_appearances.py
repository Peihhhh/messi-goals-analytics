"""
Tạo hai bảng `matches` và `appearances` từ StatsBomb Open Data.

Phạm vi: mọi trận giải vô địch quốc gia mà Messi có tên trong đội hình,
gồm La Liga (Barcelona, 2004/05 đến 2020/21) và Ligue 1 (PSG, 2021/22 đến 2022/23).

Cách chạy (từ thư mục gốc của repo):
    pip install statsbombpy pandas
    python src/build_matches_and_appearances.py

Kết quả:
    data/processed/matches.csv
    data/processed/appearances.csv
    data/processed/messi.db   (SQLite, chứa cùng hai bảng trên)
"""

import sqlite3
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

# statsbombpy in cảnh báo "credentials were not supplied" mỗi lần gọi.
# Dữ liệu mở không cần tài khoản, nên tắt cảnh báo này cho gọn.
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------

MESSI_PLAYER_ID = 5503

# Mã giải đấu trong StatsBomb và CLB của Messi ở giải đó
LEAGUE_CONFIGS = [
    {
        "competition_id": 11,
        "competition_name": "La Liga",
        "club": "Barcelona",
        "first_season_start_year": 2004,
        "last_season_start_year": 2020,
    },
    {
        "competition_id": 7,
        "competition_name": "Ligue 1",
        "club": "Paris Saint-Germain",
        "first_season_start_year": 2021,
        "last_season_start_year": 2022,
    },
]

MESSI_BIRTH_YEAR = 1987
REGULATION_MINUTES = 90

OUTPUT_DIRECTORY = Path("data/processed")


# ---------------------------------------------------------------------------
# Bước 1: tìm các mùa giải cần tải
# ---------------------------------------------------------------------------

def get_season_start_year(season_name):
    """'2011/2012' -> 2011"""
    return int(season_name.split("/")[0])


def find_seasons_to_download():
    """Trả về danh sách các mùa (mỗi mùa là một dict) nằm trong phạm vi bài."""
    all_competitions = sb.competitions()
    seasons_to_download = []

    for league_config in LEAGUE_CONFIGS:
        is_this_league = (
            all_competitions["competition_id"] == league_config["competition_id"]
        )
        league_seasons = all_competitions[is_this_league]

        for _, season_row in league_seasons.iterrows():
            season_start_year = get_season_start_year(season_row["season_name"])

            # Bỏ các mùa ngoài phạm vi, ví dụ La Liga 1973/74 hay Ligue 1 2015/16
            is_in_range = (
                league_config["first_season_start_year"]
                <= season_start_year
                <= league_config["last_season_start_year"]
            )
            if not is_in_range:
                continue

            seasons_to_download.append(
                {
                    "competition_id": league_config["competition_id"],
                    "competition_name": league_config["competition_name"],
                    "club": league_config["club"],
                    "season_id": int(season_row["season_id"]),
                    "season_name": season_row["season_name"],
                    "season_start_year": season_start_year,
                }
            )

    seasons_to_download.sort(key=lambda season: season["season_start_year"])
    return seasons_to_download


# ---------------------------------------------------------------------------
# Bước 2: đọc đội hình từng trận để tính số phút của Messi
# ---------------------------------------------------------------------------

def clock_string_to_minute(clock_string):
    """'67:31' -> 67. StatsBomb ghi thời điểm theo đồng hồ trận đấu (phút:giây)."""
    minute_part = clock_string.split(":")[0]
    return int(minute_part)


def compute_minutes_from_positions(position_spells):
    """
    Tính số phút thi đấu từ danh sách "position spells" của một cầu thủ.

    Mỗi spell là một khoảng thời gian cầu thủ chơi ở một vị trí, có 'from' và 'to'.
    Nếu cầu thủ đổi vị trí giữa trận thì có nhiều spell nối tiếp nhau, nên chỉ cần
    lấy thời điểm bắt đầu của spell đầu và thời điểm kết thúc của spell cuối.

    Quy ước (giống Transfermarkt): không tính bù giờ, một trận tối đa 90 phút.
    """
    if len(position_spells) == 0:
        # Có tên trong danh sách đăng ký nhưng không ra sân
        return 0

    first_spell = position_spells[0]
    last_spell = position_spells[-1]

    start_minute = clock_string_to_minute(first_spell["from"])

    if last_spell["to"] is None:
        # 'to' rỗng nghĩa là chơi đến hết trận
        end_minute = REGULATION_MINUTES
    else:
        end_minute = clock_string_to_minute(last_spell["to"])

    # Cắt các mốc trong thời gian bù giờ về phút 90
    start_minute = min(start_minute, REGULATION_MINUTES)
    end_minute = min(end_minute, REGULATION_MINUTES)

    minutes_played = end_minute - start_minute

    # Vào sân ở phút bù giờ vẫn được tính tối thiểu 1 phút
    minutes_played = max(minutes_played, 1)
    return minutes_played


def get_messi_appearance(match_id, club):
    """
    Trả về dict mô tả lần ra sân của Messi trong một trận,
    hoặc None nếu anh ấy không ra sân.
    """
    lineups_by_team = sb.lineups(match_id=match_id)

    if club not in lineups_by_team:
        return None

    club_lineup = lineups_by_team[club]
    messi_rows = club_lineup[club_lineup["player_id"] == MESSI_PLAYER_ID]

    if len(messi_rows) == 0:
        return None

    position_spells = messi_rows.iloc[0]["positions"]
    minutes_played = compute_minutes_from_positions(position_spells)

    if minutes_played == 0:
        return None

    first_spell = position_spells[0]
    last_spell = position_spells[-1]

    return {
        "match_id": match_id,
        "started": int(first_spell["start_reason"] == "Starting XI"),
        "minutes_played": minutes_played,
        "starting_position": first_spell["position"],
        "end_reason": last_spell["end_reason"],
    }


# ---------------------------------------------------------------------------
# Bước 3: ghép thành hai bảng
# ---------------------------------------------------------------------------

def build_match_row(statsbomb_match, season):
    """Chuyển một dòng trận đấu của StatsBomb thành một dòng của bảng `matches`."""
    club = season["club"]
    is_home_match = statsbomb_match["home_team"] == club

    if is_home_match:
        opponent = statsbomb_match["away_team"]
        venue = "home"
        goals_for = statsbomb_match["home_score"]
        goals_against = statsbomb_match["away_score"]
    else:
        opponent = statsbomb_match["home_team"]
        venue = "away"
        goals_for = statsbomb_match["away_score"]
        goals_against = statsbomb_match["home_score"]

    return {
        "match_id": int(statsbomb_match["match_id"]),
        "date": statsbomb_match["match_date"],
        "season": season["season_name"],
        "age": season["season_start_year"] - MESSI_BIRTH_YEAR,
        "competition": season["competition_name"],
        "match_week": int(statsbomb_match["match_week"]),
        "club": club,
        "opponent": opponent,
        "venue": venue,
        "home_score": int(statsbomb_match["home_score"]),
        "away_score": int(statsbomb_match["away_score"]),
        "goals_for": int(goals_for),
        "goals_against": int(goals_against),
    }


def build_tables():
    """Tải dữ liệu và trả về (matches_table, appearances_table)."""
    match_rows = []
    appearance_rows = []

    for season in find_seasons_to_download():
        statsbomb_matches = sb.matches(
            competition_id=season["competition_id"],
            season_id=season["season_id"],
        )

        number_of_appearances_this_season = 0

        for _, statsbomb_match in statsbomb_matches.iterrows():
            # Chỉ giữ các trận của CLB mà Messi đang thi đấu
            club_is_playing = season["club"] in (
                statsbomb_match["home_team"],
                statsbomb_match["away_team"],
            )
            if not club_is_playing:
                continue

            match_id = int(statsbomb_match["match_id"])
            appearance = get_messi_appearance(match_id, season["club"])

            # Bỏ các trận Messi không ra sân
            if appearance is None:
                continue

            match_rows.append(build_match_row(statsbomb_match, season))
            appearance_rows.append(appearance)
            number_of_appearances_this_season += 1

        print(
            f"{season['competition_name']} {season['season_name']}: "
            f"{number_of_appearances_this_season} trận"
        )

    matches_table = pd.DataFrame(match_rows).sort_values("date").reset_index(drop=True)
    appearances_table = pd.DataFrame(appearance_rows)

    # Sắp xếp appearances theo cùng thứ tự ngày với matches
    match_order = matches_table[["match_id"]]
    appearances_table = match_order.merge(appearances_table, on="match_id")

    return matches_table, appearances_table


# ---------------------------------------------------------------------------
# Bước 4: lưu và kiểm tra
# ---------------------------------------------------------------------------

def save_tables(matches_table, appearances_table):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    matches_table.to_csv(OUTPUT_DIRECTORY / "matches.csv", index=False)
    appearances_table.to_csv(OUTPUT_DIRECTORY / "appearances.csv", index=False)

    database_connection = sqlite3.connect(OUTPUT_DIRECTORY / "messi.db")
    matches_table.to_sql(
        "matches", database_connection, if_exists="replace", index=False
    )
    appearances_table.to_sql(
        "appearances", database_connection, if_exists="replace", index=False
    )
    database_connection.close()


def print_season_summary(matches_table, appearances_table):
    """In số trận và số phút theo mùa để đối chiếu với Transfermarkt."""
    merged_table = matches_table.merge(appearances_table, on="match_id")

    season_summary = merged_table.groupby(
        ["season", "age", "competition"], as_index=False
    ).agg(
        appearances=("match_id", "count"),
        starts=("started", "sum"),
        minutes=("minutes_played", "sum"),
    )

    print()
    print(season_summary.to_string(index=False))
    print()
    print(f"Tổng số trận: {len(merged_table)}")
    print(f"Tổng số phút: {merged_table['minutes_played'].sum()}")


def main():
    matches_table, appearances_table = build_tables()

    # Mỗi trận chỉ được xuất hiện đúng một lần
    assert matches_table["match_id"].is_unique
    assert appearances_table["match_id"].is_unique
    assert len(matches_table) == len(appearances_table)

    save_tables(matches_table, appearances_table)
    print_season_summary(matches_table, appearances_table)


if __name__ == "__main__":
    main()
