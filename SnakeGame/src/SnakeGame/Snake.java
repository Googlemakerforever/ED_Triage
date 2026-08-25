package SnakeGame;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.event.KeyEvent;
import java.util.ArrayList;

import utilities.GDV5;

public class Snake {
    private static ArrayList<Tiles> body = new ArrayList<Tiles>();
    private Color color = Color.orange;
    private static Tiles head;
    private int score = 0;

    public Snake() {
        body = new ArrayList<Tiles>();
        int size = 20;
        int col = 30;
        int row = 20;
        int x = col * size;
        int y = row * size;

        for (int i = 0; i < 5; i++) {
            body.add(new Tiles(x, y, size, size, color, row, col));
            x += size;
            col++;
        }

        head = body.get(0);
    }

    public void move(Board b) {
        int row;
        int col;
        int direction;

        for (Tiles t : body) {
            row = t.getRow();
            col = t.getCol();
            if (row >= 0 && row < b.getGrid().length && col >= 0 && col < b.getGrid()[0].length) {
                direction = b.getGrid()[row][col].getDirection();
                t.setDirection(direction);
                t.move();
            }
        }
    }

    public void turn(Board b) {
        int row;
        int col;
        int direction;

        if (GDV5.KeysPressed[KeyEvent.VK_DOWN]) {
            head.setDirection(1);
        }
        if (GDV5.KeysPressed[KeyEvent.VK_RIGHT]) {
            head.setDirection(2);
        }
        if (GDV5.KeysPressed[KeyEvent.VK_UP]) {
            head.setDirection(3);
        }
        if (GDV5.KeysPressed[KeyEvent.VK_LEFT]) {
            head.setDirection(0);
        }

        row = head.getRow();
        col = head.getCol();
        direction = head.getDirection();
        if (row >= 0 && row < b.getGrid().length && col >= 0 && col < b.getGrid()[0].length) {
            b.getGrid()[row][col].setDirection(direction);
        }
    }

    public void draw(Graphics2D pb) {
        pb.setColor(color);
        for (Tiles t : body) {
            t.fill(pb);
        }
    }

    public boolean snakeHitWall(Board b) {
        int headX = head.getCol();
        int headY = head.getRow();
        if (headX < 0 || headX >= b.getGrid()[0].length || headY < 0 || headY >= b.getGrid().length) {
            return true;
        }
        return false;
    }

    public int getHeadX() {
        return head.getCol();
    }

    public int getHeadY() {
        return head.getRow();
    }

    public static void grow() {
        Tiles tail = body.get(body.size() - 1);
        int direction = tail.getDirection();
        int row = tail.getRow();
        int col = tail.getCol();

        if (direction == 0) {
            col++;
        } else if (direction == 1) {
            row++;
        } else if (direction == 2) {
            col--;
        } else if (direction == 3) {
            row--;
        }

        int x = col * 20;
        int y = row * 20;
        Tiles newTile = new Tiles(x, y, 20, 20, Color.ORANGE, row, col);
        newTile.setDirection(direction);
        body.add(newTile);
    }

    public void addScore(int points) {
        score += points;
    }

    public int getScore() {
        return score;
    }
}
