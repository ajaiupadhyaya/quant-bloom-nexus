import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

interface D3LineChartProps {
  data: { x: string | number; y: number }[];
  width?: number;
  height?: number;
  color?: string;
  title?: string;
  xLabel?: string;
  yLabel?: string;
  style?: React.CSSProperties;
}

export const D3LineChart: React.FC<D3LineChartProps> = ({
  data,
  width = 600,
  height = 320,
  color = '#1a73e8',
  title,
  xLabel,
  yLabel,
  style = {},
}) => {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!data || data.length === 0) return;
    const svg = d3.select(ref.current);
    svg.selectAll('*').remove();

    // Margins
    const margin = { top: 40, right: 30, bottom: 50, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Convert x values to strings for consistent handling
    const processedData = data.map(d => ({ ...d, x: String(d.x) }));

    // Scales
    const x = d3
      .scalePoint()
      .domain(processedData.map((d) => d.x))
      .range([0, innerWidth]);
    const y = d3
      .scaleLinear()
      .domain([d3.min(data, (d) => d.y) ?? 0, d3.max(data, (d) => d.y) ?? 1])
      .nice()
      .range([innerHeight, 0]);

    // Axes
    const xAxis = (g: d3.Selection<SVGGElement, unknown, null, undefined>) =>
      g
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x).tickSizeOuter(0))
        .call((g) => g.selectAll('.domain').attr('stroke', '#bbb'))
        .call((g) => g.selectAll('text').attr('font-size', 12).attr('fill', '#888'));
    const yAxis = (g: d3.Selection<SVGGElement, unknown, null, undefined>) =>
      g
        .call(d3.axisLeft(y).ticks(6))
        .call((g) => g.selectAll('.domain').attr('stroke', '#bbb'))
        .call((g) => g.selectAll('text').attr('font-size', 12).attr('fill', '#888'));

    // Gridlines
    svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)
      .call((g) =>
        g
          .append('g')
          .attr('class', 'grid')
          .selectAll('line')
          .data(y.ticks(6))
          .join('line')
          .attr('x1', 0)
          .attr('x2', innerWidth)
          .attr('y1', (d) => y(d))
          .attr('y2', (d) => y(d))
          .attr('stroke', '#eee')
      );

    // Line
    const line = d3
      .line<{ x: string; y: number }>()
      .x((d) => x(d.x) ?? 0)
      .y((d) => y(d.y));

    svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)
      .append('path')
      .datum(processedData)
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 2.5)
      .attr('d', line)
      .attr('filter', 'url(#glow)');

    // Dots and tooltips
    const tooltip = d3
      .select(ref.current?.parentElement)
      .selectAll('.d3-tooltip')
      .data([null])
      .join('div')
      .attr('class', 'd3-tooltip')
      .style('position', 'absolute')
      .style('pointer-events', 'none')
      .style('background', 'rgba(255,255,255,0.95)')
      .style('border', '1px solid #bbb')
      .style('border-radius', '4px')
      .style('padding', '6px 10px')
      .style('font-size', '13px')
      .style('color', '#222')
      .style('box-shadow', '0 2px 8px rgba(0,0,0,0.07)')
      .style('display', 'none');

    svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)
      .selectAll('circle')
      .data(processedData)
      .join('circle')
      .attr('cx', (d) => x(d.x) ?? 0)
      .attr('cy', (d) => y(d.y))
      .attr('r', 4)
      .attr('fill', color)
      .attr('opacity', 0.7)
      .on('mouseover', function (event, d) {
        d3.select(this).attr('r', 7);
        tooltip
          .style('display', 'block')
          .html(`<strong>${xLabel || 'X'}:</strong> ${d.x}<br/><strong>${yLabel || 'Y'}:</strong> ${d.y}`)
          .style('left', event.pageX + 10 + 'px')
          .style('top', event.pageY - 28 + 'px');
      })
      .on('mouseout', function () {
        d3.select(this).attr('r', 4);
        tooltip.style('display', 'none');
      });

    // Axes
    svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)
      .call(xAxis);
    svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)
      .call(yAxis);

    // Title
    if (title) {
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', margin.top / 2)
        .attr('text-anchor', 'middle')
        .attr('font-size', 20)
        .attr('font-family', 'Georgia, serif')
        .attr('font-weight', 'bold')
        .attr('fill', '#222')
        .text(title);
    }
    // X label
    if (xLabel) {
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', height - 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', 14)
        .attr('fill', '#888')
        .text(xLabel);
    }
    // Y label
    if (yLabel) {
      svg
        .append('text')
        .attr('transform', `rotate(-90)`)
        .attr('x', -height / 2)
        .attr('y', 18)
        .attr('text-anchor', 'middle')
        .attr('font-size', 14)
        .attr('fill', '#888')
        .text(yLabel);
    }
    // Artistic glow filter
    svg
      .append('defs')
      .append('filter')
      .attr('id', 'glow')
      .append('feGaussianBlur')
      .attr('stdDeviation', 2.5)
      .attr('result', 'coloredBlur');
  }, [data, width, height, color, title, xLabel, yLabel]);

  return (
    <div style={{ position: 'relative', width, height, ...style }}>
      <svg ref={ref} width={width} height={height} />
    </div>
  );
}; 