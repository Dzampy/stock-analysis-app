# Debug Guide pro nefungující tlačítka

## Jak zjistit, proč tlačítka nefungují:

1. **Otevři Developer Console (F12 nebo Cmd+Option+I)**
2. **Zkontroluj, zda vidíš tyto zprávy:**
   - `🚀 Script started loading...`
   - `✅ navigateToSection is now available on window`
   - `✅ Functions exported to window scope`

3. **Zkus kliknout na tlačítko a sleduj konzoli:**
   - Pokud vidíš `navigateToSection not available` → funkce není exportovaná
   - Pokud vidíš `Uncaught ReferenceError: navigateToSection is not defined` → syntax error v JavaScriptu
   - Pokud nevidíš žádnou chybu → problém může být v CSS (z-index, pointer-events)

4. **Zkontroluj v konzoli ručně:**
   ```javascript
   typeof window.navigateToSection  // mělo by být "function"
   typeof window.toggleSidebar      // mělo by být "function"
   window.debugCheck()              // mělo by vrátit true
   ```

5. **Zkontroluj, zda tlačítka nejsou překrytá:**
   - V Elements panelu najdi tlačítko
   - Zkontroluj computed styles: `pointer-events` (nemělo by být `none`)
   - Zkontroluj `z-index` (mělo by být dostatečně vysoké)

## Možné problémy:

1. **Syntax Error v JavaScriptu** → celý script se neprovede
2. **Funkce nejsou exportované** → console.log ukáže chybu
3. **CSS blokuje klikání** → pointer-events: none nebo z-index problém
4. **Event listener není připojený** → onclick atribut se neprovede

## Rychlé řešení:

Pokud nic z výše uvedeného nepomůže, zkus:
1. Tvrdý refresh stránky (Ctrl+Shift+R nebo Cmd+Shift+R)
2. Vymazat cache prohlížeče
3. Zkontrolovat, zda Render deployment proběhl úspěšně

