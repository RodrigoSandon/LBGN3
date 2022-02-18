"""print("Way 2")
        x = list(df[cell_x])
        y = list(df[cell_y])

        start_1 = time.time()
        cost = dtw(x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=3)
        print("Alignment cost: {:.4f}".format(cost))

        end_1 = time.time()
        print(f"Time taken: {end_1 - start_1}")

        print("Way 3")

        start_2 = time.time()
        cost = dtw(x, y, global_constraint="itakura", itakura_max_slope=2.)
        print("Alignment cost: {:.4f}".format(cost))

        end_2 = time.time()
        print(f"Time taken: {end_2 - start_2}")"""
