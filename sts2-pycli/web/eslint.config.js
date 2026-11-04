import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
    },
  },
  {
    // Dev-only harness pages (not shipped): each is a Vite entry that renders
    // its own page, so it has no exports to hot-swap and an edit reloads the
    // page — which is what a harness wants. The rule has nothing to protect.
    files: ['src/gallery.tsx', 'src/cardskin.tsx', 'src/boardpreview.tsx'],
    rules: { 'react-refresh/only-export-components': 'off' },
  },
)
