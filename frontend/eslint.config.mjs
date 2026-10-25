// ESLint 9 flat config. `next lint` was removed in Next 16, so the old
// "lint": "next lint" script had silently stopped working.
import nextVitals from "eslint-config-next/core-web-vitals"
import nextTypescript from "eslint-config-next/typescript"

const config = [
  ...nextVitals,
  ...nextTypescript,
  {
    ignores: [".next/**", "node_modules/**", "next-env.d.ts"],
  },
  {
    // Generated shadcn/ui primitives: kept as shipped, not linted to our rules.
    files: ["components/ui/**", "hooks/**"],
    rules: {
      "@typescript-eslint/no-empty-object-type": "off",
      "@typescript-eslint/no-unused-vars": "off",
      "react-hooks/purity": "off",
      "react-hooks/set-state-in-effect": "off",
    },
  },
]

export default config
