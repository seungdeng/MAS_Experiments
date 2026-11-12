def solution(maps):

    def bfs(x, y):
        queue = [(x, y)]
        visited[x][y] = True
        distance = 1

        while queue:
            for _ in range(len(queue)):
                curr_x, curr_y = queue.pop(0)

                if curr_x == len(maps) - 1 and curr_y == len(maps[0]) - 1:
                    return distance

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = curr_x + dx, curr_y + dy

                    if 0 <= nx < len(maps) and 0 <= ny < len(maps[0]) and not visited[nx][ny] and maps[nx][ny] == 1:
                        visited[nx][ny] = True
                        queue.append((nx, ny))

            distance += 1

        return -1