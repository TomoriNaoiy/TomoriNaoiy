import os

from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash

from helpers import apology, login_required, lookup, usd

# Configure application
app = Flask(__name__)

# Custom filter
app.jinja_env.filters["usd"] = usd

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///finance.db")


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
@login_required
def index():
    """Show portfolio of stocks"""
    user_id = session["user_id"]

    # 获取用户持仓
    holdings = db.execute(
        "SELECT symbol, shares FROM portfolios WHERE user_id = ?", user_id
    )

    # 获取用户现金
    cash = db.execute("SELECT cash FROM users WHERE id = ?", user_id)[0]["cash"]

    total = cash  # 总资产 = 现金 + 股票市值

    # 获取当前股票价格并计算总值
    for holding in holdings:
        stock = lookup(holding["symbol"])
        holding["name"] = stock["name"]
        holding["price"] = stock["price"]
        holding["total"] = holding["shares"] * stock["price"]
        total += holding["total"]

    return render_template("index.html", holdings=holdings, cash=cash, total=total)



@app.route("/buy", methods=["GET", "POST"])
@login_required
def buy():
    """Buy shares of stock"""
    if request.method == "POST":
        symbol = request.form.get("symbol")
        shares = request.form.get("shares")

        # 检查 symbol 和 shares 是否存在
        if not symbol:
            return apology("missing symbol", 400)
        if not shares:
            return apology("missing shares", 400)

        # 检查 shares 是正整数
        try:
            shares = int(shares)
            if shares <= 0:
                return apology("invalid shares", 400)
        except ValueError:
            return apology("shares must be an integer", 400)

        # 查找股票价格
        stock = lookup(symbol.upper())
        if stock is None:
            return apology("invalid symbol", 400)

        price = stock["price"]
        total_cost = price * shares
        user_id = session["user_id"]

        # 查用户现金余额
        cash = db.execute("SELECT cash FROM users WHERE id = ?", user_id)[0]["cash"]
        if cash < total_cost:
            return apology("not enough cash", 400)

        # 更新用户现金
        db.execute("UPDATE users SET cash = cash - ? WHERE id = ?", total_cost, user_id)

        # 更新持仓记录（如果已有该股票，则加；没有则插入）
        rows = db.execute(
            "SELECT shares FROM portfolios WHERE user_id = ? AND symbol = ?",
            user_id, stock["symbol"]
        )
        if len(rows) == 0:
            # 没持有过，插入新记录
            db.execute(
                "INSERT INTO portfolios (user_id, symbol, shares) VALUES (?, ?, ?)",
                user_id, stock["symbol"], shares
            )
        else:
            # 持有过，更新数量
            db.execute(
                "UPDATE portfolios SET shares = shares + ? WHERE user_id = ? AND symbol = ?",
                shares, user_id, stock["symbol"]
            )

        # 记录交易历史
        db.execute(
            "INSERT INTO history (user_id, symbol, shares, price, type) VALUES (?, ?, ?, ?, ?)",
            user_id, stock["symbol"], shares, price, "BUY"
        )

        # 跳转首页
        return redirect("/")

    else:
        return render_template("buy.html")



@app.route("/history")
@login_required
def history():
    user_id = session["user_id"]

    # 查询当前用户的所有交易记录
    rows = db.execute(
        "SELECT symbol, shares, price, type, timestamp FROM history WHERE user_id = ? ORDER BY timestamp DESC",
        user_id
    )

    return render_template("history.html", rows=rows)



@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 403)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 403)

        # Query database for username
        rows = db.execute(
            "SELECT * FROM users WHERE username = ?", request.form.get("username")
        )

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(
            rows[0]["hash"], request.form.get("password")
        ):
            return apology("invalid username and/or password", 403)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/quote", methods=["GET", "POST"])
@login_required
def quote():
    """Get stock quote."""
    if request.method=='POST':
        symbol=request.form.get("symbol")
        if not symbol:
            return apology("where is your symbol?")
        stock = lookup(symbol.upper())

        if stock is None:
            return apology("invalid symbol", 400)
        return render_template("quoted.html",name=stock["name"], symbol=stock["symbol"], price=stock["price"])
    else:
        return render_template("quote.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        if not username:
            return apology("where is your username?", 400)
        if not password:
            return apology("where is your password?", 400)
        if password != confirmation:
            return apology("please provide same password", 400)

        # Hash the password
        hash_pw = generate_password_hash(password)

        # 尝试插入用户，并捕获重复用户名异常
        try:
            db.execute("INSERT INTO users (username, hash) VALUES (?, ?)", username, hash_pw)
        except:
            return apology("username already exists", 400)

        # 获取新注册用户的 ID
        user_row = db.execute("SELECT id FROM users WHERE username = ?", username)
        session["user_id"] = user_row[0]["id"]

        return redirect("/")

    else:
        return render_template("register.html")



@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    user_id = session["user_id"]

    if request.method == "POST":
        symbol = request.form.get("symbol")
        shares = request.form.get("shares")

        # 校验输入
        if not symbol:
            return apology("must provide symbol", 400)
        if not shares:
            return apology("must provide shares", 400)
        try:
            shares = int(shares)
            if shares <= 0:
                return apology("shares must be positive", 400)
        except ValueError:
            return apology("shares must be integer", 400)

        # 检查持仓是否足够
        rows = db.execute(
            "SELECT shares FROM portfolios WHERE user_id = ? AND symbol = ?",
            user_id, symbol
        )
        if not rows:
            return apology("you don't own this stock", 400)
        owned_shares = rows[0]["shares"]
        if shares > owned_shares:
            return apology("not enough shares", 400)

        # 获取股票当前价格
        stock = lookup(symbol.upper())
        if stock is None:
            return apology("invalid symbol", 400)

        price = stock["price"]
        total = price * shares

        # 增加现金
        db.execute("UPDATE users SET cash = cash + ? WHERE id = ?", total, user_id)

        # 减少持仓（或删除）
        if shares == owned_shares:
            db.execute(
                "DELETE FROM portfolios WHERE user_id = ? AND symbol = ?",
                user_id, symbol
            )
        else:
            db.execute(
                "UPDATE portfolios SET shares = shares - ? WHERE user_id = ? AND symbol = ?",
                shares, user_id, symbol
            )

        # 添加交易历史
        db.execute(
            "INSERT INTO history (user_id, symbol, shares, price, type) VALUES (?, ?, ?, ?, ?)",
            user_id, symbol.upper(), -shares, price, "SELL"
        )

        return redirect("/")

    else:
        # 查询当前持仓的 symbol 列表，用于下拉选择
        symbols = db.execute(
            "SELECT symbol FROM portfolios WHERE user_id = ?", user_id
        )
        return render_template("sell.html", symbols=[row["symbol"] for row in symbols])
