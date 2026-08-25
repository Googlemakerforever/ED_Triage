const products = [
  {
    id: "honeycrisp",
    name: "Honeycrisp",
    category: "snacking",
    price: 2.2,
    rating: 4.9,
    emoji: "🍎",
    description: "Explosive crunch with balanced sweetness."
  },
  {
    id: "granny-smith",
    name: "Granny Smith",
    category: "baking",
    price: 1.6,
    rating: 4.7,
    emoji: "🍏",
    description: "Bright tart flavor perfect for pies."
  },
  {
    id: "fuji",
    name: "Fuji",
    category: "snacking",
    price: 1.9,
    rating: 4.8,
    emoji: "🍎",
    description: "Sweet and juicy all-day snack apple."
  },
  {
    id: "pink-lady",
    name: "Pink Lady",
    category: "snacking",
    price: 2.1,
    rating: 4.6,
    emoji: "🍎",
    description: "Floral aroma with crisp bite and tang."
  },
  {
    id: "braeburn",
    name: "Braeburn",
    category: "baking",
    price: 1.8,
    rating: 4.5,
    emoji: "🍏",
    description: "Firm texture that holds shape in baking."
  },
  {
    id: "orchard-gift-box",
    name: "Orchard Gift Box",
    category: "gift",
    price: 24,
    rating: 5,
    emoji: "🎁",
    description: "Handpicked premium mix, 12-count box."
  }
];

const promoCodes = {
  APPLE10: 0.1,
  HARVEST15: 0.15
};

const state = {
  cart: loadJSON("apple_cart", {}),
  promo: loadJSON("apple_promo", null),
  orders: loadJSON("apple_orders", [])
};

const refs = {
  grid: document.getElementById("product-grid"),
  search: document.getElementById("search"),
  sort: document.getElementById("sort"),
  category: document.getElementById("category"),
  cartButton: document.getElementById("cart-button"),
  cartCount: document.getElementById("cart-count"),
  cartDrawer: document.getElementById("cart-drawer"),
  closeCart: document.getElementById("close-cart"),
  cartItems: document.getElementById("cart-items"),
  totals: document.getElementById("totals"),
  promoInput: document.getElementById("promo"),
  applyPromo: document.getElementById("apply-promo"),
  checkout: document.getElementById("checkout"),
  overlay: document.getElementById("overlay"),
  checkoutDialog: document.getElementById("checkout-dialog"),
  checkoutForm: document.getElementById("checkout-form"),
  cancelCheckout: document.getElementById("cancel-checkout"),
  successDialog: document.getElementById("success-dialog"),
  orderSummary: document.getElementById("order-summary"),
  closeSuccess: document.getElementById("close-success")
};

function loadJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

function saveState() {
  localStorage.setItem("apple_cart", JSON.stringify(state.cart));
  localStorage.setItem("apple_promo", JSON.stringify(state.promo));
  localStorage.setItem("apple_orders", JSON.stringify(state.orders));
}

function money(v) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD"
  }).format(v);
}

function getFilteredProducts() {
  const query = refs.search.value.trim().toLowerCase();
  const category = refs.category.value;
  const sort = refs.sort.value;

  let list = products.filter((p) => {
    const matchesQuery =
      !query || p.name.toLowerCase().includes(query) || p.description.toLowerCase().includes(query);
    const matchesCategory = category === "all" || p.category === category;
    return matchesQuery && matchesCategory;
  });

  if (sort === "price-asc") list = list.sort((a, b) => a.price - b.price);
  if (sort === "price-desc") list = list.sort((a, b) => b.price - a.price);
  if (sort === "rating-desc") list = list.sort((a, b) => b.rating - a.rating);

  return list;
}

function renderProducts() {
  const list = getFilteredProducts();
  refs.grid.innerHTML = "";

  if (!list.length) {
    refs.grid.innerHTML = '<p>No apples match your filters.</p>';
    return;
  }

  list.forEach((p, idx) => {
    const card = document.createElement("article");
    card.className = "card";
    card.style.animationDelay = `${idx * 45}ms`;
    card.innerHTML = `
      <div class="image" aria-hidden="true">${p.emoji}</div>
      <div class="card-content">
        <h3>${p.name}</h3>
        <p class="meta">${p.description}</p>
        <p class="meta">${"★".repeat(Math.round(p.rating))} (${p.rating.toFixed(1)})</p>
        <div class="price-row">
          <span class="price">${money(p.price)} / lb</span>
          <button class="add-btn" data-id="${p.id}">Add</button>
        </div>
      </div>
    `;
    refs.grid.appendChild(card);
  });
}

function cartEntries() {
  return Object.entries(state.cart)
    .map(([id, qty]) => ({ product: products.find((p) => p.id === id), qty }))
    .filter((item) => item.product && item.qty > 0);
}

function getTotals() {
  const subtotal = cartEntries().reduce((sum, item) => sum + item.product.price * item.qty, 0);
  const discountRate = state.promo ? promoCodes[state.promo] || 0 : 0;
  const discount = subtotal * discountRate;
  const taxed = subtotal - discount;
  const tax = taxed * 0.0825;
  const shipping = subtotal > 0 && subtotal < 20 ? 4.99 : 0;
  const total = taxed + tax + shipping;

  return { subtotal, discount, tax, shipping, total };
}

function renderCart() {
  const items = cartEntries();
  refs.cartItems.innerHTML = "";

  if (!items.length) {
    refs.cartItems.innerHTML = '<p class="meta">Your cart is empty.</p>';
  }

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "cart-item";
    row.innerHTML = `
      <div>
        <strong>${item.product.name}</strong>
        <div class="meta">${money(item.product.price)} / lb</div>
      </div>
      <div>
        <div class="qty">
          <button data-change="-1" data-id="${item.product.id}" aria-label="Decrease quantity">-</button>
          <span>${item.qty}</span>
          <button data-change="1" data-id="${item.product.id}" aria-label="Increase quantity">+</button>
        </div>
      </div>
    `;
    refs.cartItems.appendChild(row);
  });

  const count = items.reduce((sum, item) => sum + item.qty, 0);
  refs.cartCount.textContent = count;

  const totals = getTotals();
  const promoLine = state.promo ? `<div>Promo (${state.promo}): -${money(totals.discount)}</div>` : "";
  refs.totals.innerHTML = `
    <div>Subtotal: ${money(totals.subtotal)}</div>
    ${promoLine}
    <div>Tax: ${money(totals.tax)}</div>
    <div>Shipping: ${money(totals.shipping)}</div>
    <hr />
    <strong>Total: ${money(totals.total)}</strong>
  `;

  saveState();
}

function toggleCart(open) {
  const show = typeof open === "boolean" ? open : !refs.cartDrawer.classList.contains("open");
  refs.cartDrawer.classList.toggle("open", show);
  refs.overlay.hidden = !show;
  refs.cartDrawer.setAttribute("aria-hidden", String(!show));
}

function addToCart(id) {
  state.cart[id] = (state.cart[id] || 0) + 1;
  renderCart();
}

function changeQty(id, change) {
  if (!state.cart[id]) return;
  state.cart[id] += change;
  if (state.cart[id] <= 0) delete state.cart[id];
  renderCart();
}

function applyPromo() {
  const code = refs.promoInput.value.trim().toUpperCase();
  if (!code) {
    state.promo = null;
    renderCart();
    return;
  }

  if (promoCodes[code]) {
    state.promo = code;
    renderCart();
  } else {
    alert("Invalid promo code. Try APPLE10 or HARVEST15.");
  }
}

function checkout() {
  if (!cartEntries().length) {
    alert("Your cart is empty.");
    return;
  }
  refs.checkoutDialog.showModal();
}

function placeOrder(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  const totals = getTotals();

  const order = {
    id: `OC-${Date.now()}`,
    name: form.get("name"),
    email: form.get("email"),
    address: form.get("address"),
    total: totals.total,
    items: cartEntries().map((e) => ({ id: e.product.id, qty: e.qty })),
    placedAt: new Date().toISOString()
  };

  state.orders.push(order);
  state.cart = {};
  state.promo = null;
  saveState();
  renderCart();

  refs.checkoutDialog.close();
  refs.orderSummary.textContent = `${order.id} confirmed for ${order.name}. Total charged: ${money(
    order.total
  )}. A receipt was sent to ${order.email}.`;
  refs.successDialog.showModal();
  event.target.reset();
}

refs.grid.addEventListener("click", (e) => {
  const button = e.target.closest(".add-btn");
  if (button) addToCart(button.dataset.id);
});

refs.cartItems.addEventListener("click", (e) => {
  const button = e.target.closest("button[data-id]");
  if (button) changeQty(button.dataset.id, Number(button.dataset.change));
});

refs.search.addEventListener("input", renderProducts);
refs.sort.addEventListener("change", renderProducts);
refs.category.addEventListener("change", renderProducts);
refs.cartButton.addEventListener("click", () => toggleCart(true));
refs.closeCart.addEventListener("click", () => toggleCart(false));
refs.overlay.addEventListener("click", () => toggleCart(false));
refs.applyPromo.addEventListener("click", applyPromo);
refs.checkout.addEventListener("click", checkout);
refs.checkoutForm.addEventListener("submit", placeOrder);
refs.cancelCheckout.addEventListener("click", () => refs.checkoutDialog.close());
refs.closeSuccess.addEventListener("click", () => refs.successDialog.close());

renderProducts();
renderCart();
