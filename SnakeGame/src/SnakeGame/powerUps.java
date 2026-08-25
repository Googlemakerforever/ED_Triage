package SnakeGame;

import java.awt.Color;
import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class powerUps {
    private int size;
    private static Random ran = new Random();
    private List<Apple> apples;
    private int lastEatenType = -1;

    private static class Apple {
        int x;
        int y;
        int type;
        Color color;

        Apple(int x, int y, int type, Color color) {
            this.x = x;
            this.y = y;
            this.type = type;
            this.color = color;
        }
    }

    public powerUps(int tSize) {
        this.size = tSize;
        apples = new ArrayList<Apple>();
        spawnNormalApple();
    }

    private void spawnApple(int forcedType) {
        int col = 1200 / size;
        int row = 800 / size;
        int minTile = 1;
        int maxTileX = col - 2;
        int maxTileY = row - 2;

        int x = (ran.nextInt(maxTileX - minTile + 1) + minTile) * size;
        int y = (ran.nextInt(maxTileY - minTile + 1) + minTile) * size;

        int type = forcedType;
        if (type < 0) {
            type = ran.nextDouble() < (2.0 / 3.0) ? 0 : 1;
        }

        Color color = type == 0 ? Color.RED : Color.GREEN;
        apples.add(new Apple(x, y, type, color));
    }

    private void spawnNormalApple() {
        spawnApple(-1);
    }

    private void spawnThreeRedApples() {
        for (int i = 0; i < 3; i++) {
            spawnApple(0);
        }
    }

    public void draw(Graphics2D win) {
        for (Apple apple : apples) {
            drawAppleSprite(win, apple.x, apple.y, size, apple.color);
        }
    }

    public static void drawAppleSprite(Graphics2D win, int x, int y, int size, Color appleColor) {
        int leftLobeX = x + size / 7;
        int rightLobeX = x + (size * 3) / 7;
        int lobeY = y + size / 4;
        int lobeSize = (size * 3) / 5;
        int bodyX = x + size / 4;
        int bodyY = y + size / 2;
        int bodyW = size / 2;
        int bodyH = size / 3;

        win.setColor(new Color(130, 92, 52));
        win.fillRect(x + size / 2 - 1, y + size / 9, 2, Math.max(2, size / 4));

        win.setColor(new Color(72, 166, 66));
        win.fillOval(x + (size * 5) / 9, y + size / 12, Math.max(4, size / 3), Math.max(3, size / 5));

        win.setColor(appleColor);
        win.fillOval(leftLobeX, lobeY, lobeSize, lobeSize);
        win.fillOval(rightLobeX, lobeY, lobeSize, lobeSize);
        win.fillOval(bodyX, bodyY, bodyW, bodyH);

        win.setColor(appleColor.darker());
        win.drawOval(leftLobeX, lobeY, lobeSize, lobeSize);
        win.drawOval(rightLobeX, lobeY, lobeSize, lobeSize);
        win.drawOval(bodyX, bodyY, bodyW, bodyH);

        win.setColor(new Color(255, 255, 255, 160));
        win.fillOval(x + size / 3, y + size / 3, Math.max(2, size / 5), Math.max(2, size / 5));
    }

    public boolean eaten(Snake s) {
        int headX = s.getHeadX();
        int headY = s.getHeadY();

        for (int i = 0; i < apples.size(); i++) {
            Apple apple = apples.get(i);
            int appleGridX = apple.x / size;
            int appleGridY = apple.y / size;
            if (headX == appleGridX && headY == appleGridY) {
                lastEatenType = apple.type;
                apples.remove(i);
                return true;
            }
        }
        return false;
    }

    public int getType() {
        return lastEatenType;
    }

    public void effect(Snake s) {
        if (lastEatenType == 0) {
            s.grow();
            s.grow();
            s.addScore(1);
        } else if (lastEatenType == 1) {
            s.grow();
            s.grow();
            s.addScore(5);
        }
    }

    public void reset() {
        if (lastEatenType == 1) {
            apples.clear();
            spawnThreeRedApples();
        } else if (apples.isEmpty()) {
            spawnNormalApple();
        }
        lastEatenType = -1;
    }

    public int getX() {
        return apples.isEmpty() ? -1 : apples.get(0).x;
    }

    public int getY() {
        return apples.isEmpty() ? -1 : apples.get(0).y;
    }
}
