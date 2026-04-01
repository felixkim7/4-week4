# import os
from playwright.sync_api import sync_playwright

def clean_cell(text):
    parts = text.split("\n")
    
    parts = [p.strip() for p in parts if p.strip()]
    
    cleaned = parts[0] if parts else ""

    # cleaned = cleaned.replace("󰌾", "")

    return cleaned.strip()

def normalize_color(color_text):
    return color_text.replace(" ", "").lower()

def get_percent_info(cell):
    p = cell.locator("p").first

    percent_text = p.inner_text().strip()
    color = normalize_color(
        p.evaluate("el => getComputedStyle(el).color")
    )

    color_map = {
        "rgb(9,133,81)": "positive",   # #098551
        "rgb(207,32,47)": "negative",  # #cf202f
        "rgb(10,11,13)": "zero",       # #0a0b0d
    }

    status = color_map.get(color, "unknown")

    
    if status == "positive" and not percent_text.startswith("+"):
        signed_percent = f"+{percent_text}"
    elif status == "negative" and not percent_text.startswith("-"):
        signed_percent = f"-{percent_text.lstrip('+')}"
    else:
        signed_percent = percent_text

    return {
        "value": percent_text,
        "status": status,
        "signed_value": signed_percent,
        "color": color,
    }

play = sync_playwright().start()
browser = play.chromium.launch(headless=False,args=["--start-maximized"])
page = browser.new_page(no_viewport=True)

selected_cols = [1, 2, 6, 7, 8]

all_page_data = []

tag_body = []

for page_num in range(1, 6):
    if page_num == 1:
        url = "https://www.coinbase.com/explore"
    else:
        url = f"https://www.coinbase.com/explore/s/all?page={page_num}"

    page.goto(url)
    page.pause()

    table = page.locator('table[data-testid="prices-table"]')

    ths = table.locator("thead th").all()
    columns = [ths[i].inner_text() for i in selected_cols]

    first_row = table.locator("tbody tr").filter(has=page.locator("td")).first

    # first_data = first_row.locator("td").all_inner_texts()

    fds = first_row.locator("td").all()
    first_data = [fds[i].inner_text() for i in selected_cols]

    print("컬럼명:", columns)
    print("첫 번째 데이터:", first_data)


    # 테이블 정리
    tag_table = table
    if page_num == 1:
        tag_header = [clean_cell(tag_table.locator("thead th").all()[i].inner_text()) for i in selected_cols]

    all_rows = tag_table.locator("tbody tr").filter(has=page.locator("td")).all()

    # rows = all_rows[1:]

    
    for row in all_rows:
        row_cells = row.locator("td").all()
        row_data = []

        for j, i in enumerate(selected_cols):
            cell = row_cells[i]

            if j == 2:
                # value = cell.locator("p").inner_text()
                percent_info = get_percent_info(cell)
                row_data.append(percent_info["signed_value"])

            else:
                value = clean_cell(cell.inner_text())
                row_data.append(value.strip())
            
            

        tag_body.append(row_data)


# page.goto("https://www.coinbase.com/explorehttps://www.coinbase.com/explore")
# page.pause()

# ##################################################
# #######         코드를 작성해주세요           #######
# ##################################################


# # 해외증시 클릭
# # page.get_by_role("link", name="해외증시").click()


# # 해외 주요지수 table 선택
# table = page.locator('table[data-testid="prices-table"]')

# # column name 추출
# # columns = table.locator("thead th").all_inner_texts()



# ths = table.locator("thead th").all()
# columns = [ths[i].inner_text() for i in selected_cols]

# # 첫 번째 데이터 행 추출
# first_row = table.locator("tbody tr").filter(has=page.locator("td")).first

# # first_data = first_row.locator("td").all_inner_texts()

# fds = first_row.locator("td").all()
# first_data = [fds[i].inner_text() for i in selected_cols]

# print("컬럼명:", columns)
# print("첫 번째 데이터:", first_data)



# # 테이블 정리
# tag_table = table
# tag_header = [clean_cell(tag_table.locator("thead th").all()[i].inner_text()) for i in selected_cols]

# all_rows = tag_table.locator("tbody tr").filter(has=page.locator("td")).all()

# # rows = all_rows[1:]

# tag_body = []
# for row in all_rows:
#     row_cells = row.locator("td").all()
#     row_data = []

#     for j, i in enumerate(selected_cols):
#         cell = row_cells[i]

#         if j == 2:
#             value = cell.locator("p").inner_text()
#         else:
#             value = clean_cell(cell.inner_text())
        
#         row_data.append(value.strip())

#     tag_body.append(row_data)

   



# 반복문으로 출력
print("\n[헤더]")
for h in tag_header:
    print(h)

print("\n[바디]")
for row in tag_body:
    print(row)



# json 출력
# import json

# dumped = json.dumps(
#     {"header": tag_header, "body": tag_body},
#     indent=2,
#     ensure_ascii=False
# )
# with open("page_2.json", "w", encoding="utf-8") as fp:
#     fp.write(dumped)

with open("crypto.txt", "w", encoding="utf-8") as fp:
    # header
    fp.write("\t".join(tag_header) + "\n")
    
    # body
    for row in tag_body:
        fp.write("\t".join(row) + "\n")


browser.close()

play.stop()
