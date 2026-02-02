---
layout: page
title: Playground
---

<script setup>
import { defineClientComponent } from 'vitepress'

const WickPlayground = defineClientComponent(() =>
  import('./.vitepress/playground/components/WickPlayground.vue')
)
</script>

# Playground

Try Wick expressions in the browser. Select a domain profile, type an expression, and see the generated code for each backend.

<WickPlayground />
