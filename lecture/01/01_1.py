try:
    # 提示使用者輸入
    user_input = input("請輸入 1~100 範圍內的整數：")
    n = int(user_input)

    # 檢查範圍是否在 1 到 100 之間
    if 1 <= n <= 100:
        # 計算 1 到 n 的累加值
        total = sum(range(1, n + 1))
        #total = (1 + n) * n // 2
        print(f"從 1 至您輸入的整數值 {n} 的累加結果為: {total}")
    else:
        print("警告：您輸入的整數不在 1~100 的有效範圍內！")

except ValueError:
    print("警告：輸入無效！請務必輸入一個「整數」。")

